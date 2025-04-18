#include "stl/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl {

bool TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool TestTaskSTL::ValidationImpl() {
  return ((task_data->inputs_count[0] == task_data->outputs_count[0]) &&
          (task_data->outputs.size() == task_data->outputs_count.size()));
}

bool TestTaskSTL::RunImpl() {
  ShellSort();
  size_t mid = input_.size() / 2;
  std::vector<int> left(input_.begin(), input_.begin() + static_cast<std::ptrdiff_t>(mid));
  std::vector<int> right(input_.begin() + static_cast<std::ptrdiff_t>(mid), input_.end());

  BatcherMerge(left, right, output_);

  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void TestTaskSTL::ShellSort() {
  int n = static_cast<int>(input_.size());
  std::vector<int> gaps;

  for (int k = 1; (1 << k) - 1 < n; ++k) {
    gaps.push_back((1 << k) - 1);
  }

  auto& input_ref = input_;

  for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
    int gap = *it;
    if (gap == 0) continue; // Гарантируем, что шаг не нулевой

    std::vector<int> groups(gap);
    std::iota(groups.begin(), groups.end(), 0);

    std::for_each(std::execution::par, groups.begin(), groups.end(), [&](int group) {
      for (int i = group + gap; i < n; i += gap) {
        int temp = input_ref[i];
        int j = i;
        while (j >= gap && input_ref[j - gap] > temp) {
          input_ref[j] = input_ref[j - gap];
          j -= gap;
        }
        input_ref[j] = temp;
      }
    });
  }
}

void TestTaskSTL::BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result) {
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

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl