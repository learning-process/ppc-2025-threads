#include "seq/opolin_d_radix_sort_betcher_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool opolin_d_radix_betcher_sort_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  size_ = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}

bool opolin_d_radix_betcher_sort_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  if (size_ <= 0 || task_data->inputs[0].empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_betcher_sort_seq::TestTaskSequential::RunImpl() {
  if (input_.empty()) {
    return;
  }
  std::vector<int> positives;
  std::vector<int> negatives;
  for (int value : input_) {
    if (value >= 0) {
      positives.push_back(value);
    } else {
      negatives.push_back(-value);
    }
  }
  int max_abs = 0;
  for (int value : input_) {
    max_abs = std::max(max_abs, std::abs(value));
  }
  int digit_count = 0;
  if (max_abs == 0) {
    digit_count = 1;
  } else {
    while (max_abs > 0) {
      max_abs /= 10;
      digit_count++;
    }
  }
  for (int place = 1; digit_count > 0; place *= 10, digit_count--) {
    if (!positives.empty()) {
      SortByDigit(positives, place);
    }
    if (!negatives.empty()) {
      SortByDigit(negatives, place);
    }
  }

  if (!negatives.empty()) {
    std::reverse(negatives.begin(), negatives.end());
  }

  output_.clear();
  for (int value : negatives) {
    output_.push_back(-value);
  }
  for (int value : positives) {
    output_.push_back(value);
  }
  BetcherMerge(output_, 0, output_.size());
  return true;
}

bool opolin_d_radix_betcher_sort_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void opolin_d_radix_betcher_sort_seq::SortByDigit(std::vector<int> &array, int digit_place) {
  const int base = 10;
  std::vector<int> result(array.size());
  std::vector<int> buckets(base, 0);

  for (int value : array) {
    int digit = (value / digit_place) % base;
    buckets[digit]++;
  }
  for (int i = 1; i < base; i++) {
    buckets[i] += buckets[i - 1];
  }
  for (int i = array.size() - 1; i >= 0; i--) {
    int digit = (array[i] / digit_place) % base;
    result[--buckets[digit]] = array[i];
  }
  array = result;
}

void opolin_d_radix_betcher_sort_seq::BetcherMerge(std::vector<int> &arr, size_t start, size_t end) {
  size_t size = end - start;
  if (size <= 1) {
    return;
  }
  size_t mid = start + size / 2;
  BetcherMerge(arr, start, mid);
  BetcherMerge(arr, mid, end);
  size_t step = (size + 1) / 2;
  while (step > 0) {
    for (size_t i = start; i + step < end; ++i) {
      size_t j = i + step;
      if (arr[i] > arr[j]) {
        std::swap(arr[i], arr[j]);
      }
    }
    step /= 2;
  }
}