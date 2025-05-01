#include "stl/solovyev_d_shell_sort_simple/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  for (int gap = static_cast<int>(input_.size()) / 2; gap > 0; gap /= 2) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([this, gap, t, num_threads]() {
        for (int i = t; i < gap; i += num_threads) {
          for (size_t f = i + gap; f < input_.size(); f += gap) {
            int val = input_[f];
            size_t j = f;
            while (j >= gap && input_[j - gap] > val) {
              input_[j] = input_[j - gap];
              j -= gap;
            }
            input_[j] = val;
          }
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
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
