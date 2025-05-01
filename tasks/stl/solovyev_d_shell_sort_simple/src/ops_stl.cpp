#include "stl/solovyev_d_shell_sort_simple/include/ops_stl.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
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

bool solovyev_d_shell_sort_simple_stl::TaskSTL::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  std::mutex mtx;
  std::condition_variable cv;
  bool ready = false;
  bool done = false;
  int current_gap = 0;
  std::vector<std::thread> threads(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([&, t]() {
      while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return ready || done; });
        if (done) return;
        int gap = current_gap;
        lock.unlock();
        for (int i = t; i < gap; i += num_threads) {
          for (size_t f = i + gap; f < input_.size(); f += gap) {
            int val = input_[f];
            size_t j = f;
            while (j >= static_cast<size_t>(gap) && input_[j - gap] > val) {
              input_[j] = input_[j - gap];
              j -= gap;
            }
            input_[j] = val;
          }
        }
      }
    });
  }
  for (int gap = static_cast<int>(input_.size()) / 2; gap > 0; gap /= 2) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      current_gap = gap;
      ready = true;
    }
    cv.notify_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
      std::lock_guard<std::mutex> lock(mtx);
      ready = false;
    }
  }
  {
    std::lock_guard<std::mutex> lock(mtx);
    done = true;
  }
  cv.notify_all();
  for (auto &th : threads) {
    if (th.joinable()) th.join();
  }
  return true;
}

bool solovyev_d_shell_sort_simple_stl::TaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
