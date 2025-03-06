#include "seq/Konstantinov_I_Sort_Batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <vector>

namespace konstantinov_i_sort_batcher_seq {

static uint64_t DoubleToKey(double d) {
  uint64_t u;
  std::memcpy(&u, &d, sizeof(d));

  if (u >> 63) {
    return ~u;
  } else {
    return u ^ 0x8000000000000000ULL;
  }
}
static double KeyToDouble(uint64_t key) {
  if (key >> 63) {
    key = key ^ 0x8000000000000000ULL;
  } else {
    key = ~key;
  }
  double d;
  std::memcpy(&d, &key, sizeof(d));
  return d;
}

static void RadixSorted(std::vector<double>& arr) {
  size_t n = arr.size();
  std::vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = DoubleToKey(arr[i]);
  }

  const int RADIX = 256;
  std::vector<uint64_t> outputKeys(n);

  for (int pass = 0; pass < 8; pass++) {
    std::vector<size_t> count(RADIX, 0);
    int shift = pass * 8;
    for (size_t i = 0; i < n; i++) {
      int byte = (keys[i] >> shift) & 0xFF;
      count[byte]++;
    }
    for (int j = 1; j < RADIX; j++) {
      count[j] += count[j - 1];
    }
    for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
      int byte = (keys[i] >> shift) & 0xFF;
      outputKeys[--count[byte]] = keys[i];
    }
    keys.swap(outputKeys);
  }

  for (size_t i = 0; i < n; i++) {
    arr[i] = KeyToDouble(keys[i]);
  }
}

void RadixSortBatcherSeq::BatcherOddEvenMerge(std::vector<double>& arr, int low, int high) {
  if (high - low <= 1) return;
  int mid = (low + high) / 2;
  RadixSortBatcherSeq::BatcherOddEvenMerge(arr, low, mid);
  RadixSortBatcherSeq::BatcherOddEvenMerge(arr, mid, high);

  for (int i = low; i < mid; ++i) {
    if (arr[i] > arr[i + mid - low]) {
      std::swap(arr[i], arr[i + mid - low]);
    }
  }
}

void RadixSortBatcherSeq::RadixSort(std::vector<double>& arr) {
  konstantinov_i_sort_batcher_seq::RadixSorted(arr);
  BatcherOddEvenMerge(arr, 0, static_cast<int>(arr.size()));
}

}  // namespace konstantinov_i_sort_batcher_seq

bool konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  mas_ = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq::RunImpl() {
  output_ = mas_;
  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq::RadixSort(output_);
  return true;
}

bool konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
