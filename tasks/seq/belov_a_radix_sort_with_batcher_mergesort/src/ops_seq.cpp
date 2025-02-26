#include "seq/belov_a_radix_sort_with_batcher_mergesort/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

using namespace std;

namespace belov_a_radix_batcher_mergesort_seq {
int RadixBatcherMergesortSequential::GetNumberDigitCapacity(Bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortSequential::Sort(vector<Bigint>& arr) {
  vector<Bigint> pos;
  vector<Bigint> neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  RadixSort(pos, false);
  RadixSort(neg, true);

  arr.clear();
  arr.reserve(neg.size() + pos.size());
  for (const auto& num : neg) {
    arr.push_back(-num);
  }
  arr.insert(arr.end(), pos.begin(), pos.end());
}

void RadixBatcherMergesortSequential::RadixSort(vector<Bigint>& arr, bool invert) {
  if (arr.empty()) {
    return;
  }

  Bigint max_val = *std::ranges::max_element(arr);
  int max_val_digit_capacity = GetNumberDigitCapacity(max_val);
  int iter = 1;

  for (Bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    CountingSort(arr, digit_place);
  }

  if (invert) {
    std::ranges::reverse(arr);
  }
}

void RadixBatcherMergesortSequential::CountingSort(vector<Bigint>& arr, Bigint digit_place) {
  vector<Bigint> output(arr.size());
  int count[10] = {};

  for (const auto& num : arr) {
    Bigint index = (num / digit_place) % 10;
    count[index]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (size_t i = arr.size() - 1; i < arr.size(); i--) {
    Bigint num = arr[i];
    Bigint index = (num / digit_place) % 10;
    output[--count[index]] = num;
  }

  std::ranges::copy(output, arr.begin());
}

bool RadixBatcherMergesortSequential::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  auto* input_array_data = reinterpret_cast<Bigint*>(task_data->inputs[0]);
  array_.assign(input_array_data, input_array_data + n_);

  return true;
}

bool RadixBatcherMergesortSequential::ValidationImpl() {
  return (task_data->inputs.size() == 1 && !(task_data->inputs_count.size() < 2) && task_data->inputs_count[0] != 0 &&
          (task_data->inputs_count[0] == task_data->inputs_count[1]) && !task_data->outputs.empty());
}

bool RadixBatcherMergesortSequential::RunImpl() {
  Sort(array_);
  return true;
}

bool RadixBatcherMergesortSequential::PostProcessingImpl() {
  std::ranges::copy(array_, reinterpret_cast<Bigint*>(task_data->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_seq
