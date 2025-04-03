#include "tbb/gusev_n_sorting_int_simple_merging/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <numeric>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

void gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::RadixSort(std::vector<int>& arr) {
  if (arr.empty()) return;

  std::vector<int> negatives;
  std::vector<int> positives;

  for (int num : arr) {
    if (num < 0)
      negatives.push_back(-num);
    else
      positives.push_back(num);
  }

  tbb::parallel_invoke(
      [&] {
        if (!negatives.empty()) {
          RadixSortForNonNegative(negatives);
          std::reverse(negatives.begin(), negatives.end());
          for (auto& num : negatives) num = -num;
        }
      },
      [&] {
        if (!positives.empty()) {
          RadixSortForNonNegative(positives);
        }
      });

  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

void gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::RadixSortForNonNegative(
    std::vector<int>& arr) {
  if (arr.empty()) return;

  int max = *std::max_element(arr.begin(), arr.end());
  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::CountingSort(std::vector<int>& arr, int exp) {
  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  tbb::enumerable_thread_specific<std::vector<int>> tl_counts([&] { return std::vector<int>(10, 0); });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()), [&](const tbb::blocked_range<size_t>& r) {
    auto& local_counts = tl_counts.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      int digit = (arr[i] / exp) % 10;
      local_counts[digit]++;
    }
  });

  for (const auto& lc : tl_counts) {
    for (int d = 0; d < 10; ++d) {
      count[d] += lc[d];
    }
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (auto i = arr.size(); i > 0; --i) {
    int digit = (arr[i - 1] / exp) % 10;
    output[--count[digit]] = arr[i - 1];
  }

  arr = output;
}

bool gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::PreProcessingImpl() {
  input_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  output_.resize(input_.size());
  return true;
}

bool gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::RunImpl() {
  RadixSort(input_);
  return true;
}

bool gusev_n_sorting_int_simple_merging_tbb::SortingIntSimpleMergingTBB::PostProcessingImpl() {
  std::copy(input_.begin(), input_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
