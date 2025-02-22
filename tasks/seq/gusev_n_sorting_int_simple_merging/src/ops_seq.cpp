#include "seq/gusev_n_sorting_int_simple_merging/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RadixSort(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  std::vector<int> negatives;
  std::vector<int> positives;

  for (int num : arr) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }

  if (!negatives.empty()) {
    RadixSortForNonNegative(negatives);
    std::reverse(negatives.begin(), negatives.end());
    for (int& num : negatives) {
      num = -num;
    }
  }

  if (!positives.empty()) {
    RadixSortForNonNegative(positives);
  }

  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

void gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RadixSortForNonNegative(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max = *std::max_element(arr.begin(), arr.end());
  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::CountingSort(std::vector<int>& arr, int exp) {
  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  for (int num : arr) {
    int digit = (num / exp) % 10;
    count[digit]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (int i = arr.size() - 1; i >= 0; i--) {
    int digit = (arr[i] / exp) % 10;
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }

  arr = output;
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  output_ = std::vector<int>(input_size);

  return true;
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RunImpl() {
  gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RadixSort(input_);
  output_ = input_;
  return true;
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
