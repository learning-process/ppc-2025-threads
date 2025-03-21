#include "seq/kovalchuk_a_shell_sort/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace kovalchuk_a_shell_sort {

ShellSortSequential::ShellSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign (input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool ShellSortSequential::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortSequential::RunImpl() {
  shellSort();
  return true;
}

void ShellSortSequential::shellSort() {
  if (input_.empty()) return;

  int n = input_.size();
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = input_[i];
      int j;
      for (j = i; j >= gap && input_[j - gap] > temp; j -= gap) {
        input_[j] = input_[j - gap];
      }
      input_[j] = temp;
    }
  }
}

bool ShellSortSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(input_.begin(), input_.end(), output_ptr);
  return true;
}

}  // namespace kovalchuk_a_shell_sort