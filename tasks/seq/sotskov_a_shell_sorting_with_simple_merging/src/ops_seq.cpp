#include "seq/sotskov_a_shell_sorting_with_simple_merging/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

std::vector<int> sotskov_a_shell_sorting_with_simple_merging_seq::ShellSort(const std::vector<int>& input_array) {
  std::vector<int> sorted_array = input_array;
  std::size_t array_size = sorted_array.size();

  std::vector<std::size_t> gap_sequence;
  std::size_t current_gap = 1;
  while (current_gap < array_size / 3) {
    gap_sequence.push_back(current_gap);
    current_gap = current_gap * 3 + 1;
  }

  for (std::size_t gap_index = gap_sequence.size() - 1; gap_index != std::size_t(-1); --gap_index) {
    std::size_t gap = gap_sequence[gap_index];
    for (std::size_t i = gap; i < array_size; ++i) {
      int current_element = sorted_array[i];
      std::size_t j = i;
      while (j >= gap && sorted_array[j - gap] > current_element) {
        sorted_array[j] = sorted_array[j - gap];
        j -= gap;
      }
      sorted_array[j] = current_element;
    }
  }

  return sorted_array;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* temp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = temp_ptr[i];
  }

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  std::size_t output_size = task_data->outputs_count[0];

  return (input_size == output_size);
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::RunImpl() {
  result_ = ShellSort(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(result_.begin(), result_.end(), output);
  return true;
}
