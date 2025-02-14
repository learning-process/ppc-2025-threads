#include "seq/sotskov_a_shell_sorting_with_simple_merging/include/ops_seq.hpp"

#include <cmath>
#include <ranges>
#include <vector>

std::vector<int> sotskov_a_shell_sorting_with_simple_merging_seq::ShellSort(const std::vector<int>& input_array) {
  std::vector<int> sorted_array = input_array;
  int array_size = sorted_array.size();

  std::vector<int> gap_sequence;
  int current_gap = 1;
  while (current_gap < array_size / 3) {
    gap_sequence.push_back(current_gap);
    current_gap = current_gap * 3 + 1;
  }

  for (int gap_index = gap_sequence.size() - 1; gap_index >= 0; --gap_index) {
    int gap = gap_sequence[gap_index];
    for (int i = gap; i < array_size; ++i) {
      int current_element = sorted_array[i];
      int j = i;
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
  int input_size = task_data->inputs_count[0];
  int output_size = task_data->outputs_count[0];

  return (input_size == output_size);
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::RunImpl() {
  result_ = ShellSort(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  int* output_ = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(result_.begin(), result_.end(), output_);
  return true;
}
