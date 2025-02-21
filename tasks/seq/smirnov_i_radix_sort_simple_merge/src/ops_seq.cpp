#include "seq/smirnov_i_radix_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  mas = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::RunImpl() {
  int longest = *std::max_element(mas.begin(), mas.end());
  int len = std::ceil(std::log10(longest + 1));
  std::vector<int> sorting(mas.size());

  for (int j = 0; j < len; j++) {
    std::vector<int> counting(10, 0);
    for (size_t i = 0; i < mas.size(); i++) {
      counting[mas[i] / int(pow(10, j)) % 10]++;
    }
    std::partial_sum(counting.begin(), counting.end(), counting.begin());
    for (int i = mas.size() - 1; i >= 0; i--) {
      int pos = counting[mas[i] / int(pow(10, j)) % 10] - 1;
      sorting[pos] = mas[i];
      counting[mas[i] / int(pow(10, j)) % 10]--;
    }
    std::swap(mas, sorting);
  }
  output_ = mas;
  printf("mas %d %d %d %d", mas[0], mas[1], mas[2], mas[3]);
  printf("output_ %d %d %d %d", output_[0], output_[1], output_[2], output_[3]);
  return true;
}

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::PostProcessingImpl() {
  printf("output_ %d %d %d %d", output_[0], output_[1], output_[2], output_[3]);
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}