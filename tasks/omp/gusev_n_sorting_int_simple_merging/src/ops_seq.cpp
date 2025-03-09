#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "omp/gusev_n_sorting_int_simple_merging/include/ops_omp.hpp"

void gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::RadixSort(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  std::vector<int> negatives;
  std::vector<int> positives;

#pragma omp parallel
  {
    std::vector<int> local_neg, local_pos;
#pragma omp for nowait
    for (int i = 0; i < arr.size(); ++i) {
      int num = arr[i];
      if (num < 0) {
        local_neg.push_back(-num);
      } else {
        local_pos.push_back(num);
      }
    }

#pragma omp critical
    {
      negatives.insert(negatives.end(), local_neg.begin(), local_neg.end());
      positives.insert(positives.end(), local_pos.begin(), local_pos.end());
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (!negatives.empty()) {
        RadixSortForNonNegative(negatives);
        std::reverse(negatives.begin(), negatives.end());
        for (auto& num : negatives) num = -num;
      }
    }
#pragma omp section
    {
      if (!positives.empty()) {
        RadixSortForNonNegative(positives);
      }
    }
  }

  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

void gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::RadixSortForNonNegative(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max_val = arr[0];
#pragma omp parallel for reduction(max : max_val)
  for (int i = 0; i < arr.size(); ++i) {
    if (arr[i] > max_val) max_val = arr[i];
  }
  int max = max_val;

  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::CountingSort(std::vector<int>& arr, int exp) {
  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

#pragma omp parallel
  {
    std::vector<int> local_count(10, 0);
#pragma omp for
    for (int i = 0; i < arr.size(); ++i) {
      int num = arr[i];
      int digit = (num / exp) % 10;
      local_count[digit]++;
    }

#pragma omp critical
    {
      for (int j = 0; j < 10; ++j) count[j] += local_count[j];
    }
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (size_t i = arr.size(); i > 0; --i) {
    int digit = (arr[i - 1] / exp) % 10;
    output[count[digit] - 1] = arr[i - 1];
    count[digit]--;
  }

  arr = output;
}

bool gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_.resize(input_.size());
  return true;
}

bool gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::RunImpl() {
  gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::RadixSort(input_);
  return true;
}

bool gusev_n_sorting_int_simple_merging_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
