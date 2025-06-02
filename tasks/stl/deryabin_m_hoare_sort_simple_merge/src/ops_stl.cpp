#include "stl/deryabin_m_hoare_sort_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void deryabin_m_hoare_sort_simple_merge_stl::HoareSort(std::vector<double>& a, size_t first, size_t last) {
  if (first >= last) {
    return;
  }
  const double x = a[(first + last) >> 1];
  double* pi = &a[first];
  double* pj = &a[last];
  do {
    while (*pi < x) {
      pi++;
    }
    while (*pj > x) {
      pj--;
    }
    const double tmp = *pi;
    *pi = *pj;
    *pj = tmp;
  } while (pi < pj);
  const size_t j = pj - a.data();
  const size_t i = pi - a.data();
  HoareSort(a, first, j);
  HoareSort(a, i + 1, last);
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  while (count != chunk_count_) {
    HoareSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  size_t chunk_count = chunk_count_;
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) - 1); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      std::inplace_merge(input_array_A_.begin() + static_cast<long>(j * min_chunk_size_ << (i + 1)),
                         input_array_A_.begin() + static_cast<long>(((j << 1 | 1) * (min_chunk_size_ << i))),
                         input_array_A_.begin() + static_cast<long>((j + 1) * min_chunk_size_ << (i + 1)));
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::RunImpl() {
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  if (chunk_count_ < num_threads) {
    chunk_count_ = 1ULL << std::bit_width(num_threads - 1);
    min_chunk_size_ = dimension_ / chunk_count_;
  }
  std::barrier sync_point(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  const size_t chunks_per_thread = chunk_count_ / num_threads;
  for (size_t i = 0; i < num_threads; ++i) {
    workers.emplace_back([&sync_point, &input_array_A_, &chunks_per_thread, &chunk_count_, &min_chunk_size_, &dimension_, i] {
      const size_t start = i * chunks_per_thread;
      const size_t end = std::min((i + 1) * chunks_per_thread, chunk_count_);
      for (size_t j = start; j < end; ++j) {
        HoareSort(input_array_A_, j * min_chunk_size_, std::min((j + 1) * min_chunk_size_ - 1, dimension_ - 1));
      }
      sync_point.arrive_and_wait();
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) - 1); ++i) {
    std::barrier sync_point(num_threads);
    workers.resize(0);
    workers.reserve(num_threads);
    const size_t merge_pairs = chunk_count_ >> (i + 1);
    const size_t pairs_per_thread = (merge_pairs + num_threads - 1) / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
      workers.emplace_back([&sync_point, &input_array_A_, &merge_pairs, &pairs_per_thread, &min_chunk_size_, i, t] {
        const size_t start = t * pairs_per_thread;
        const size_t end = std::min((t + 1) * pairs_per_thread, merge_pairs);
        for (size_t j = start; j < end; ++j) {
          std::inplace_merge(
            input_array_A_.begin() + static_cast<long>(j * min_chunk_size_ << (i + 1)),
            input_array_A_.begin() + static_cast<long>(((j << 1 | 1) * (min_chunk_size_ << i))),
            input_array_A_.begin() + static_cast<long>((j + 1) * min_chunk_size_ << (i + 1)));
        }
        sync_point.arrive_and_wait();
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
