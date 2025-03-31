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
  std::vector<int> positives;
  std::vector<int> negatives;
  int pos_count = 0;
  int neg_count = 0;
#pragma omp parallel
  {
    int local_pos_count = 0;
    int local_neg_count = 0;
#pragma omp for nowait
    for (int i = 0; i < size_; i++) {
      if (input_[i] >= 0) {
        local_pos_count++;
      } else {
        local_neg_count++;
      }
    }
#pragma omp critical
    {
      pos_count += local_pos_count;
      neg_count += local_neg_count;
    }
  }

  positives.resize(pos_count);
  negatives.resize(neg_count);

  int pos_index = 0;
  int neg_index = 0;
#pragma omp parallel
  {
    std::vector<int> local_pos;
    std::vector<int> local_neg;
#pragma omp for nowait
    for (int i = 0; i < size_; i++) {
      if (input_[i] >= 0) {
        local_pos.push_back(input_[i]);
      } else {
        local_neg.push_back(-input_[i]);
      }
    }
#pragma omp critical
    {
      int start_pos = pos_index;
      pos_index += local_pos.size();
      for (size_t j = 0; j < local_pos.size(); j++) {
        positives[start_pos + j] = local_pos[j];
      }
      int start_neg = neg_index;
      neg_index += local_neg.size();
      for (size_t j = 0; j < local_neg.size(); j++) {
        negatives[start_neg + j] = local_neg[j];
      }
    }
  }
  int max_abs = 0;
#pragma omp parallel
  {
    int local_max = 0;
#pragma omp for nowait
    for (int i = 0; i < size_; i++) {
      int abs_val = std::abs(input_[i]);
      if (abs_val > local_max) {
        local_max = abs_val;
      }
    }
#pragma omp critical
    {
      if (local_max > max_abs) {
        max_abs = local_max;
      }
    }
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
#pragma omp parallel sections
  {
#pragma omp section
    {
      for (int place = 1; digit_count > 0; place *= 10, digit_count--) {
        if (!positives.empty()) {
          SortByDigit(positives, place);
        }
      }
    }
#pragma omp section
    {
      for (int place = 1; digit_count > 0; place *= 10, digit_count--) {
        if (!negatives.empty()) {
          SortByDigit(negatives, place);
        }
      }
    }
  }

  if (!negatives.empty()) {
    std::reverse(negatives.begin(), negatives.end());
    for (size_t i = 0; i < negatives.size(); i++) {
      negatives[i] = -negatives[i];
    }
  }

  output_.resize(size_);
  int neg_size = negatives.size();
  int pos_size = positives.size();
#pragma omp parallel
  {
#pragma omp for nowait
    for (int i = 0; i < neg_size; i++) {
      output_[i] = negatives[i];
    }
#pragma omp for nowait
    for (int i = 0; i < pos_size; i++) {
      output_[neg_size + i] = positives[i];
    }
  }
  BatcherOddEvenMerge(output_, 0, size_);
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

void polin_d_radix_batcher_sort_omp::BatcherOddEvenMerge(std::vector<int> &array, int start, int n) {
  if (n <= 1) return;

  int m = n / 2;
  BatcherOddEvenMerge(array, start, m);
  BatcherOddEvenMerge(array, start + m, n - m);

#pragma omp parallel for
  for (int i = start; i < start + m; i++) {
    if (i + m < start + n) {
      if (array[i] > array[i + m]) {
        std::swap(array[i], array[i + m]);
      }
    }
  }
}