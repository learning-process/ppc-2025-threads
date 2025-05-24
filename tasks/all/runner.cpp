#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <atomic>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/global_control.h"

namespace {
void FlushStreams() {
  std::fflush(stdout);
  std::fflush(stderr);
}
}  // namespace

class UnreadMessagesDetector : public ::testing::EmptyTestEventListener {
 public:
  UnreadMessagesDetector(boost::mpi::communicator com) : com_(std::move(com)) {}

  void OnTestEnd(const ::testing::TestInfo& test_info) override {
    com_.barrier();
    if (const auto msg = com_.iprobe(boost::mpi::any_source, boost::mpi::any_tag)) {
      fprintf(
          stderr,
          "[  PROCESS %d  ] [  FAILED  ] %s.%s: MPI message queue has an unread message from process %d with tag %d\n",
          com_.rank(), test_info.test_suite_name(), test_info.name(), msg->source(), msg->tag());
      exit(2);
    }
    com_.barrier();
  }

 private:
  boost::mpi::communicator com_;
};

class HangDetector : public ::testing::EmptyTestEventListener {
 public:
  HangDetector(boost::mpi::communicator com, std::chrono::milliseconds timeout)
      : com_(std::move(com)), timeout_(timeout) {}

  void Watchdog() {
    do {
      {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait_for(lock, timeout_, [&] { return !should_run_.load(); });
      }

      const auto* test_info = test_info_.load();
      if (test_info == nullptr) {
        continue;
      }

      const auto elapsed = std::chrono::high_resolution_clock::now() - test_startup_timestamp_.load();
      if (elapsed > timeout_) {
        FlushStreams();
        fprintf(stderr, "[  PROCESS %d  ] [  FAILED  ] %s.%s: timed out - presumably deadlock detected\n", com_.rank(),
                test_info->test_suite_name(), test_info->name());
        std::fflush(stderr);
        abort();
      }
    } while (should_run_.load());
  }

  void OnEnvironmentsSetUpEnd(const ::testing::UnitTest&) override {
    should_run_ = true;
    watchdog_thread_ = std::thread(&HangDetector::Watchdog, this);
  }

  void OnEnvironmentsTearDownStart(const ::testing::UnitTest&) override {
    {
      std::lock_guard<std::mutex> lock(m_);
      should_run_ = false;
    }
    cv_.notify_all();
    watchdog_thread_.join();
  }

  void OnTestStart(const ::testing::TestInfo& test_info) override {
    com_.barrier();

    test_info_ = &test_info;
    test_startup_timestamp_ = std::chrono::high_resolution_clock::now();

    com_.barrier();
  }

  void OnTestEnd(const ::testing::TestInfo&) override {
    com_.barrier();

    {
      std::lock_guard<std::mutex> lock(m_);
      test_info_ = nullptr;
    }
    cv_.notify_all();

    com_.barrier();
  }

 private:
  boost::mpi::communicator com_;
  std::chrono::milliseconds timeout_;

  std::mutex m_;
  std::condition_variable cv_;

  std::atomic<const ::testing::TestInfo*> test_info_;
  std::atomic<std::chrono::high_resolution_clock::time_point> test_startup_timestamp_;
  std::atomic<bool> should_run_;

  std::thread watchdog_thread_;
};

class StreamsFlusher : public ::testing::EmptyTestEventListener {
 public:
  void OnTestEnd(const ::testing::TestInfo&) override { FlushStreams(); }
};

class WorkerTestFailurePrinter : public ::testing::EmptyTestEventListener {
 public:
  WorkerTestFailurePrinter(std::shared_ptr<::testing::TestEventListener> base, boost::mpi::communicator com)
      : base_(std::move(base)), com_(std::move(com)) {}

  void OnTestEnd(const ::testing::TestInfo& test_info) override {
    if (test_info.result()->Passed()) {
      return;
    }
    PrintProcessRank();
    base_->OnTestEnd(test_info);
  }

  void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
    if (test_part_result.passed() || test_part_result.skipped()) {
      return;
    }
    PrintProcessRank();
    base_->OnTestPartResult(test_part_result);
  }

 private:
  void PrintProcessRank() const { printf(" [  PROCESS %d  ] ", com_.rank()); }

  std::shared_ptr<::testing::TestEventListener> base_;
  boost::mpi::communicator com_;
};

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // Limit the number of threads in TBB
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, ppc::util::GetPPCNumThreads());

  ::testing::InitGoogleTest(&argc, argv);

  auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0 && (argc < 2 || argv[1] != std::string("--print-workers"))) {
    auto* listener = listeners.Release(listeners.default_result_printer());
    listeners.Append(new WorkerTestFailurePrinter(std::shared_ptr<::testing::TestEventListener>(listener), world));
  }
  listeners.Append(new StreamsFlusher);
  listeners.Append(new HangDetector(world, std::chrono::seconds(30)));
  listeners.Append(new UnreadMessagesDetector(world));

  return RUN_ALL_TESTS();
}
