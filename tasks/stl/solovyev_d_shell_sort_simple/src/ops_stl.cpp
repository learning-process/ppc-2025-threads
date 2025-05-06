#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"
#include "stl/solovyev_d_shell_sort_simple/include/ops_stl.hpp"

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
  int local_gap = -1;
  while (true) {
    std::unique_lock lock(m_);
    cv_.wait(lock, [&] { return ready_ || done_; });
    if (done_) {
      return;
    }
    if (local_gap == gap_) {
      continue;
    }
    local_gap = gap_;
    lock.unlock();
    for (int i = t; i < local_gap; i += num_threads_) {
      for (size_t f = i + local_gap; f < input_.size(); f += local_gap) {
        int val = input_[f];
        size_t j = f;
        while (j >= static_cast<size_t>(local_gap) && input_[j - local_gap] > val) {
          input_[j] = input_[j - local_gap];
          j -= local_gap;
        }
        input_[j] = val;
      }
    }
    {
      // std::lock_guard<std::mutex> cout_lock(cout_mutex_);
      // std::cout<<"Thread "<<t<<" is done!"<<std::endl;
    }
    if (++threads_completed_ == num_threads_) {
      cv_done_.notify_one();
    }
  }
}

bool solovyev_d_shell_sort_simple_stl::TaskSTL::RunImpl() {
  num_threads_ = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads_);
  for (int t = 0; t < num_threads_; ++t) {
    threads[t] = std::thread(&TaskSTL::ThreadWorker, this, t);
    done_ = false;
  }
  for (gap_ = static_cast<int>(input_.size()) / 2; gap_ > 0; gap_ /= 2) {
    {
      std::lock_guard<std::mutex> lock(m_);
      ready_ = true;
    }
    {
      // std::lock_guard<std::mutex> cout_lock(cout_mutex_);
      // std::cout<<"Gap "<<gap_<<" is starting!"<<std::endl;
    }
    cv_.notify_all();
    std::unique_lock lock(m_);
    cv_done_.wait(lock, [&] { return threads_completed_ == num_threads_; });
    threads_completed_ = 0;
    {
      // std::lock_guard<std::mutex> cout_lock(cout_mutex_);
      // std::cout<<"Gap "<<gap_<<" is done!"<<std::endl;
    }
    ready_ = false;
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
