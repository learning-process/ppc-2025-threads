#include "seq/sotskov_a_shell_sorting_with_simple_merging/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void sotskov_a_shell_sorting_with_simple_merging_seq::SimpleMerge(std::vector<int>& arr, std::size_t left,
                                                                  std::size_t mid, std::size_t right) {
  std::inplace_merge(arr.begin() + static_cast<std::ptrdiff_t>(left),
                     arr.begin() + static_cast<std::ptrdiff_t>(mid) + 1,
                     arr.begin() + static_cast<std::ptrdiff_t>(right) + 1);
}

void sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  std::size_t array_size = arr.size();
  std::vector<std::size_t> gap_sequence;
  std::size_t current_gap = 1;
  while (current_gap < array_size) {
    gap_sequence.push_back(current_gap);
    current_gap = current_gap * 3 + 1;
  }

  for (std::size_t gap_index = gap_sequence.size(); gap_index-- > 0;) {
    std::size_t gap = gap_sequence[gap_index];
    for (std::size_t i = gap; i < array_size; ++i) {
      int current_element = arr[i];
      std::size_t j = i;
      while (j >= gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
  }

  if (std::ranges::is_sorted(arr.begin(), arr.end())) {
    return;
  }

  for (std::size_t size = 1; size < array_size; size *= 2) {
    for (std::size_t left = 0; left + size < array_size; left += 2 * size) {
      std::size_t mid = left + size - 1;
      std::size_t right = std::min(left + (2 * size) - 1, array_size - 1);
      SimpleMerge(arr, left, mid, right);
    }
  }
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* temp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::ranges::copy(temp_ptr, temp_ptr + task_data->inputs_count[0], input_.begin());

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  std::size_t output_size = task_data->outputs_count[0];

  return (input_size == output_size);
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::RunImpl() {
  ShellSortWithSimpleMerging(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_.begin(), input_.end(), output);
  return true;
}
