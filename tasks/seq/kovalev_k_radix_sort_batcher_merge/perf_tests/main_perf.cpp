#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

const long long int MinLL = std::numeric_limits<long long>::lowest(), MaxLL = std::numeric_limits<long long>::max();

TEST(kovalev_k_radix_sort_batcher_merge_seq, test_pipeline_run) {
  const unsigned int length = 10;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<long long int> in(length, alpha);
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge>(taskSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  long long int *tmp = reinterpret_cast<long long int *>(out.data());
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (tmp[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_seq, test_task_run) {
  const unsigned int length = 5000000;
  std::vector<long long int> in(length);
  std::vector<long long int> out(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(MinLL, MaxLL);
  std::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> etalon(in);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge>(taskSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  std::sort(etalon.begin(), etalon.end(), [](long long int a, long long int b) { return a < b; });
  auto *tmp = reinterpret_cast<long long int *>(out.data());
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (tmp[i] != etalon[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}