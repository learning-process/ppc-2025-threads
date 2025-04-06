#include "omp/opolin_d_radix_sort_betcher_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  return true;
}

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::RunImpl() {
  int num_threads = omp_get_max_threads();
  int block_size = (size_ + num_threads - 1) / num_threads;

  std::vector<int> starts;
  std::vector<int> ends;
  for (int i = 0; i < num_threads; i++) {
    int start = i * block_size;
    int end = std::min(start + block_size, size_);
    if (start < end) {
      starts.push_back(start);
      ends.push_back(end);
    }
  }
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(starts.size()); i++) {
    int start = starts[i];
    int end = ends[i];
    std::vector<int> local_input(input_.begin() + start, input_.begin() + end);
    std::vector<int> positives;
    std::vector<int> negatives;

    for (int val : local_input) {
      if (val >= 0) {
        positives.push_back(val);
      } else {
        negatives.push_back(-val);
      }
    }
    int max_abs = 0;
    for (int val : local_input) {
      max_abs = std::max(max_abs, std::abs(val));
    }
    int digit_count = (max_abs == 0) ? 1 : 0;
    while (max_abs > 0) {
      max_abs /= 10;
      digit_count++;
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
      for (size_t j = 0; j < negatives.size(); j++) {
        negatives[j] = -negatives[j];
      }
    }
    std::vector<int> sorted_local;
    sorted_local.insert(sorted_local.end(), negatives.begin(), negatives.end());
    sorted_local.insert(sorted_local.end(), positives.begin(), positives.end());
    std::copy(sorted_local.begin(), sorted_local.end(), input_.begin() + start);
  }
  while (starts.size() > 1) {
    int merge_pairs = starts.size() / 2;
    std::vector<int> new_starts(merge_pairs);
    std::vector<int> new_ends(merge_pairs);
#pragma omp parallel for
    for (int i = 0; i < merge_pairs; i++) {
      int idx = i * 2;
      int start = starts[idx];
      int mid = ends[idx];
      int end = ends[idx + 1];
      int n1 = mid - start;
      int n2 = end - mid;
      int n = std::max(n1, n2);
      int p = 1;
      while (p < n) {
        p *= 2;
      }
      BatcherOddEvenMerge(input_, start, p, 1);
      new_starts[i] = start;
      new_ends[i] = end;
    }
    if (starts.size() % 2 == 1) {
      new_starts.push_back(starts.back());
      new_ends.push_back(ends.back());
    }
    starts = new_starts;
    ends = new_ends;
  }
  output_ = input_;
  return true;
}

bool opolin_d_radix_batcher_sort_omp::RadixBatcherSortTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void opolin_d_radix_batcher_sort_omp::SortByDigit(std::vector<int> &array, int digit_place) {
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
  for (int i = static_cast<int>(array.size() - 1); i >= 0; i--) {
    int digit = (array[i] / digit_place) % base;
    result[--buckets[digit]] = array[i];
  }
  array = result;
}

void opolin_d_radix_batcher_sort_omp::BatcherOddEvenMerge(std::vector<int> &array, int start, int n, int step) {
  if (n > 1) {
    int m = n / 2;
    BatcherOddEvenMerge(array, start, m, 2 * step);
    BatcherOddEvenMerge(array, start + step, m, 2 * step);
#pragma omp parallel for
    for (int i = start + step; i < start + n * step - step; i += 2 * step) {
      CompEx(array, i, i + step);
    }
  }
}

void opolin_d_radix_batcher_sort_omp::CompEx(std::vector<int> &array, int i, int j) {
  if (i < array.size() && j < array.size() && array[i] > array[j]) {
    std::swap(array[i], array[j]);
  }
}