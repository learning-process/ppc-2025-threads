#include "all/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_mpi.hpp"

#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {

bool TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool TestTaskMPI::ValidationImpl() {
  return ((task_data->inputs_count[0] == task_data->outputs_count[0]) &&
          (task_data->outputs.size() == task_data->outputs_count.size()));
}

bool TestTaskMPI::RunImpl() {
  size_t mid = input_.size() / 2;
  std::vector<int> left(input_.begin(), input_.begin() + static_cast<std::ptrdiff_t>(mid));
  std::vector<int> right(input_.begin() + static_cast<std::ptrdiff_t>(mid), input_.end());

#pragma omp parallel sections
  {
#pragma omp section
    { ShellSort(left); }
#pragma omp section
    { ShellSort(right); }
  }

  BatcherMerge(left, right, output_);
  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void TestTaskMPI::ShellSort(std::vector<int>& arr) {
  int n = static_cast<int>(arr.size());
  std::vector<int> gaps;
  for (int k = 1; (1 << k) - 1 < n; ++k) {
    gaps.push_back((1 << k) - 1);
  }
  for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
    int gap = *it;
#pragma omp parallel for
    for (int i = gap; i < n; ++i) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
}

void TestTaskMPI::BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result) {
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result[k++] = left[i++];
    } else {
      result[k++] = right[j++];
    }
  }
  while (i < left.size()) {
    result[k++] = left[i++];
  }
  while (j < right.size()) {
    result[k++] = right[j++];
  }
}

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi
