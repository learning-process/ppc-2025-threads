#include "stl/solovyev_d_shell_sort_simple/include/ops_stl.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool solovyev_d_shell_sort_simple_stl::TaskSTL::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  return true;
}

bool solovyev_d_shell_sort_simple_stl::TaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void solovyev_d_shell_sort_simple_stl::TaskSTL::ThreadWorker(int t) {
  while (true) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return ready_ || done_; });
    if (done_) {
      return;
    }
    for (int i = t; i < gap_; i += num_threads_) {
      for (size_t f = i + gap_; f < input_.size(); f += gap_) {
        int val = input_[f];
        size_t j = f;
        while (j >= static_cast<size_t>(gap_) && input_[j - gap_] > val) {
          input_[j] = input_[j - gap_];
          j -= gap_;
        }
        input_[j] = val;
      }
    }
  }
}

bool solovyev_d_shell_sort_simple_stl::TaskSTL::RunImpl() {
  num_threads_ = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads_);
  for (int t = 0; t < num_threads_; ++t) {
    threads[t] = std::thread(&TaskSTL::ThreadWorker, this, t);
  }
  for (gap_ = static_cast<int>(input_.size()) / 2; gap_ > 0; gap_ /= 2) {
    {
      std::lock_guard<std::mutex> lock(m_);
      ready_ = true;
    }
    cv_.notify_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
      std::lock_guard<std::mutex> lock(m_);
      ready_ = false;
    }
  }
  {
    std::lock_guard<std::mutex> lock(m_);
    done_ = true;
  }
  cv_.notify_all();
  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }
  return true;
}

bool solovyev_d_shell_sort_simple_stl::TaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
