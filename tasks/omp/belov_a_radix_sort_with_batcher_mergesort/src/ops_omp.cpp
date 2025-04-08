#include "omp/belov_a_radix_sort_with_batcher_mergesort/include/ops_omp.hpp"

#include <cmath>

#include "core/util/include/util.hpp"

using namespace std;

namespace belov_a_radix_batcher_mergesort_omp {

int RadixBatcherMergesortParallel::GetNumberDigitCapacity(Bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortParallel::Sort(std::span<Bigint> arr) {
  vector<Bigint> pos;
  vector<Bigint> neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  RadixSort(pos, false);
  RadixSort(neg, true);

  size_t index = 0;

  for (const auto& num : neg) {
    arr[index++] = -num;
  }

  for (const auto& num : pos) {
    arr[index++] = num;
  }
}

void RadixBatcherMergesortParallel::RadixSort(vector<Bigint>& arr, bool invert) {
  if (arr.empty()) {
    return;
  }

  Bigint max_val = *std::max_element(arr.begin(), arr.end());
  int max_val_digit_capacity = GetNumberDigitCapacity(max_val);
  int iter = 1;

  for (Bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    CountingSort(arr, digit_place);
  }

  if (invert) {
    std::reverse(arr.begin(), arr.end());
  }
}

void RadixBatcherMergesortParallel::CountingSort(vector<Bigint>& arr, Bigint digit_place) {
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

  std::copy(output.begin(), output.end(), arr.begin());
}

void RadixBatcherMergesortParallel::SortParallel(vector<Bigint>& arr) {
  if (arr.empty()) {
    return;
  }

  int num_threads = ppc::util::GetPPCNumThreads();
  size_t chunk_size = arr.size() / num_threads;

#pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();
    size_t start = thread_id * chunk_size;
    size_t end = (thread_id == num_threads - 1) ? arr.size() : start + chunk_size;

    std::span<Bigint> local_span(arr.data() + start, end - start);
    Sort(local_span);
  }
}

void RadixBatcherMergesortParallel::BatcherMergeParallel(vector<Bigint>& arr, int num_threads) {
  size_t n = arr.size();

  if (n == 0) {
    return;
  }

  num_threads = std::max(1, num_threads);

  size_t chunk_size = n / num_threads;            // if n < num_threads, chunk_size = 0
  size_t step = std::max<size_t>(chunk_size, 1);  // guarantee that step >= 1 (to avoid division by zero)

  for (; step < n; step *= 2) {
#pragma omp parallel for
    for (size_t i = 0; i < n - step; i += 2 * step) {
      size_t left = i;
      size_t right = i + step;
      size_t end = std::min(i + 2 * step, n);

      std::inplace_merge(arr.begin() + static_cast<int64_t>(left), arr.begin() + static_cast<int64_t>(right),
                         arr.begin() + static_cast<int64_t>(end));
    }
  }
}

vector<Bigint> RadixBatcherMergesortParallel::OddEvenMerge(const vector<Bigint>& left, const vector<Bigint>& right) {
  vector<Bigint> merged(left.size() + right.size());
  std::merge(left.begin(), left.end(), right.begin(), right.end(), merged.begin());

  for (size_t i = 1; i < merged.size(); i += 2) {
    if (i + 1 < merged.size() && merged[i] > merged[i + 1]) {
      std::swap(merged[i], merged[i + 1]);
    }
  }
  return merged;
}

bool RadixBatcherMergesortParallel::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  auto* input_array_data = reinterpret_cast<Bigint*>(task_data->inputs[0]);
  array_.assign(input_array_data, input_array_data + n_);

  return true;
}

bool RadixBatcherMergesortParallel::ValidationImpl() {
  return (task_data->inputs.size() == 1 && !(task_data->inputs_count.size() < 2) && task_data->inputs_count[0] != 0 &&
          (task_data->inputs_count[0] == task_data->inputs_count[1]) && !task_data->outputs.empty());
}

bool RadixBatcherMergesortParallel::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  SortParallel(array_);
  BatcherMergeParallel(array_, num_threads);

  return true;
}

bool RadixBatcherMergesortParallel::PostProcessingImpl() {
  copy(array_.begin(), array_.end(), reinterpret_cast<Bigint*>(task_data->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_omp