#include "omp/kovalchuk_a_shell_sort_omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalchuk_a_shell_sort_omp {

ShellSortOMP::ShellSortOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortOMP::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool ShellSortOMP::ValidationImpl() {
  return !task_data->inputs_count.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortOMP::RunImpl() {
  ShellSort();
  return true;
}

void ShellSortOMP::ShellSort() {
  if (input_.empty()) return;

  std::vector<int>& local_input = input_;
  const int n = static_cast<int>(local_input.size());

  for (int gap = n / 2; gap > 0; gap /= 2) {
#pragma omp parallel for default(none) shared(local_input) firstprivate(gap, n)
    for (int i = gap; i < n; ++i) {
      int temp = local_input[i];
      int j = i;
      for (; j >= gap && local_input[j - gap] > temp; j -= gap) {
        local_input[j] = local_input[j - gap];
      }
      local_input[j] = temp;
    }
  }
}

bool ShellSortOMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(input_.begin(), input_.end(), output_ptr);
  return true;
}

}  // namespace kovalchuk_a_shell_sort_omp