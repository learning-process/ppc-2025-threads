#include "omp/korovin_n_qsort_batcher_omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
#include <span>
#include <vector>

int korovin_n_qsort_batcher_omp::TestTaskOpenMP::GetRandomIndex(int low, int high) {
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void korovin_n_qsort_batcher_omp::TestTaskOpenMP::QuickSort(std::vector<int>& arr, int low, int high, int depth) {
  if (low >= high) {
    return;
  }

  int partition_index = GetRandomIndex(low, high);
  int partition_value = arr[partition_index];

  auto partition_iter = std::partition(arr.begin() + low, arr.begin() + high + 1,
                                       [partition_value](const int& elem) { return elem <= partition_value; });
  auto mid_iter = std::partition(arr.begin() + low, partition_iter,
                                 [partition_value](const int& elem) { return elem < partition_value; });

  int i = static_cast<int>(std::distance(arr.begin(), mid_iter));
  int j = static_cast<int>(std::distance(arr.begin(), partition_iter) - 1);

  int max_depth = static_cast<int>(std::log2(omp_get_num_threads())) + 1;

  if (depth < max_depth) {
#ifdef _MSC_VER
#pragma omp parallel sections
    {
#pragma omp section
      { QuickSort(arr, low, i - 1, depth + 1); }
#pragma omp section
      { QuickSort(arr, j + 1, high, depth + 1); }
    }
#else
#pragma omp task shared(arr)
    QuickSort(arr, low, i - 1, depth + 1);
#pragma omp task shared(arr)
    QuickSort(arr, j + 1, high, depth + 1);
#pragma omp taskwait
#endif
  } else {
    QuickSort(arr, low, i - 1, depth + 1);
    QuickSort(arr, j + 1, high, depth + 1);
  }
}

bool korovin_n_qsort_batcher_omp::TestTaskOpenMP::InPlaceMerge(std::vector<int>& arr, const BlockRange& a,
                                                               const BlockRange& b, std::vector<int>& buffer) {
  bool changed = false;

  std::span<int> span_a{arr.begin() + a.start, static_cast<size_t>(a.length)};
  std::span<int> span_b{arr.begin() + b.start, static_cast<size_t>(b.length)};

  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while (i < span_a.size() && j < span_b.size()) {
    if (span_a[i] <= span_b[j]) {
      buffer[k++] = span_a[i++];
    } else {
      changed = true;
      buffer[k++] = span_b[j++];
    }
  }
  while (i < span_a.size()) {
    buffer[k++] = span_a[i++];
  }
  while (j < span_b.size()) {
    changed = true;
    buffer[k++] = span_b[j++];
  }
  std::ranges::copy(buffer.begin(), buffer.begin() + a.length, span_a.begin());
  std::ranges::copy(buffer.begin() + a.length, buffer.begin() + a.length + b.length, span_b.begin());

  return changed;
}

std::vector<korovin_n_qsort_batcher_omp::BlockRange> korovin_n_qsort_batcher_omp::TestTaskOpenMP::PartitionBlocks(
    int n, int p) {
  std::vector<BlockRange> blocks;
  blocks.reserve(p);
  int chunk_size = n / p;
  int remainder = n % p;
  int start = 0;
  for (int i = 0; i < p; i++) {
    int size = chunk_size + (i < remainder ? 1 : 0);
    blocks.push_back({start, size});
    start += size;
  }
  return blocks;
}

void korovin_n_qsort_batcher_omp::TestTaskOpenMP::OddEvenMerge(std::vector<BlockRange>& blocks) {
  if (blocks.size() <= 1) {
    return;
  }
  int p = static_cast<int>(blocks.size());
  int max_iters = p * 2;
  int max_block_len = 0;
  for (const auto& b : blocks) {
    max_block_len = std::max(max_block_len, b.length);
  }
  int buffer_size = max_block_len * 2;

#pragma omp parallel
  {
    std::vector<int> buffer(buffer_size);
#pragma omp single
    {
      for (int iter = 0; iter < max_iters; iter++) {
        bool changed_global = false;
#pragma omp parallel for schedule(static) reduction(|| : changed_global)
        for (int b = iter % 2; b < p; b += 2) {
          if (b + 1 < p) {
            bool changed_local = InPlaceMerge(input_, blocks[b], blocks[b + 1], buffer);
            changed_global = changed_global || changed_local;
          }
        }
        if (!changed_global) {
          break;
        }
      }
    }
  }
}

bool korovin_n_qsort_batcher_omp::TestTaskOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  return true;
}

bool korovin_n_qsort_batcher_omp::TestTaskOpenMP::ValidationImpl() {
  return (!task_data->inputs.empty()) && (!task_data->outputs.empty()) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool korovin_n_qsort_batcher_omp::TestTaskOpenMP::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n <= 1) {
    return true;
  }
  int num_threads = omp_get_max_threads();
  int p = std::max(num_threads / 2, 1);
  std::vector<BlockRange> blocks = PartitionBlocks(n, p);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; i++) {
    int start = blocks[i].start;
    int end = start + blocks[i].length - 1;
    QuickSort(input_, start, end, 0);
  }
  OddEvenMerge(blocks);
  return true;
}

bool korovin_n_qsort_batcher_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
