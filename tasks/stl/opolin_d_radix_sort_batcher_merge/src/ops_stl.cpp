#include "stl/opolin_d_radix_sort_batcher_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  unsigned_data_.resize(size_);
  return true;
}

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  if (size_ <= 1) {
    output_ = input_;
    return true;
  }
  size_t unum_threads = static_cast<size_t>(num_threads);
  unum_threads = std::min((size_t)unum_threads, (static_cast<size_t>(size_) + 1) / 2);
  if (unum_threads == 0) {
    unum_threads = 1;
  }
  ParallelProcessRange(size_, unum_threads, [this](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      unsigned_data_[i] = IntToUnsigned(input_[i]);
    }
  });

  size_t block_size = (size_ + unum_threads - 1) / unum_threads;
  size_t actual_num_blocks = (size_ + block_size - 1) / block_size;
  std::vector<std::function<void()>> sort_tasks;

  for (unsigned int i = 0; i < actual_num_blocks; ++i) {
    size_t start = i * block_size;
    size_t end = std::min(start + block_size, static_cast<size_t>(size_));
    if (start < end) {
      sort_tasks.emplace_back(
          [this, start, end]() { RadixSortLSD(unsigned_data_.begin() + start, unsigned_data_.begin() + end); });
    }
  }
  if (!sort_tasks.empty()) {
    ParallelRunTasks(sort_tasks, unum_threads);
  }
  IterativeOddEvenBlockMerge(unsigned_data_.begin(), unsigned_data_.end(), actual_num_blocks, unum_threads);
  ParallelProcessRange(static_cast<size_t>(size_), unum_threads, [this](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      output_[i] = UnsignedToInt(unsigned_data_[i]);
    }
  });
  return true;
}

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

uint32_t opolin_d_radix_batcher_sort_stl::IntToUnsigned(int value) { return static_cast<uint32_t>(value) ^ (1u << 31); }

int opolin_d_radix_batcher_sort_stl::UnsignedToInt(uint32_t value) { return static_cast<int>(value ^ (1u << 31)); }

void opolin_d_radix_batcher_sort_stl::ParallelProcessRange(size_t total_size, unsigned int num_threads,
                                                           const std::function<void(size_t start, size_t end)>& func) {
  if (total_size == 0 || num_threads == 0) {
    return;
  }
  unsigned int actual_threads = std::min(num_threads, static_cast<unsigned int>(total_size));
  if (actual_threads == 0) {
    actual_threads = 1;
  }
  std::vector<std::thread> threads;
  threads.reserve(actual_threads);
  size_t block_size = (total_size + actual_threads - 1) / actual_threads;

  for (unsigned int i = 0; i < actual_threads; ++i) {
    size_t start = i * block_size;
    size_t end = std::min(start + block_size, total_size);
    if (start < end) {
      threads.emplace_back(func, start, end);
    }
  }
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void opolin_d_radix_batcher_sort_stl::ParallelRunTasks(const std::vector<std::function<void()>>& tasks,
                                                       unsigned int num_threads) {
  if (tasks.empty() || num_threads == 0) {
    return;
  }
  unsigned int actual_threads = std::min(num_threads, static_cast<unsigned int>(tasks.size()));
  if (actual_threads == 0) {
    actual_threads = 1;
  }
  std::vector<std::thread> threads;
  threads.reserve(actual_threads);
  std::atomic<size_t> task_idx(0);

  for (unsigned int i = 0; i < actual_threads; ++i) {
    threads.emplace_back([&]() {
      size_t current_task;
      while ((current_task = task_idx.fetch_add(1, std::memory_order_relaxed)) < tasks.size()) {
        tasks[current_task]();
      }
    });
  }
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void opolin_d_radix_batcher_sort_stl::RadixSortLSD(std::vector<uint32_t>::iterator begin,
                                                   std::vector<uint32_t>::iterator end) {
  size_t n = std::distance(begin, end);
  if (n <= 1) {
    return;
  }
  const int RADIX = 256;
  const int NUM_PASSES = sizeof(uint32_t);

  std::vector<uint32_t> buffer(n);
  std::vector<size_t> count(RADIX);

  for (int pass = 0; pass < NUM_PASSES; ++pass) {
    int shift = pass * 8;
    std::fill(count.begin(), count.end(), 0);
    for (auto it = begin; it != end; ++it) {
      count[(*it >> shift) & (RADIX - 1)]++;
    }
    size_t cumulative_sum = 0;
    for (size_t i = 0; i < RADIX; ++i) {
      size_t current_count = count[i];
      count[i] = cumulative_sum;
      cumulative_sum += current_count;
    }
    for (auto it = std::make_reverse_iterator(end); it != std::make_reverse_iterator(begin); ++it) {
      buffer[--count[(*it >> shift) & (RADIX - 1)]] = *it;
    }
    std::copy(buffer.begin(), buffer.end(), begin);
  }
}

void opolin_d_radix_batcher_sort_stl::IterativeOddEvenBlockMerge(std::vector<uint32_t>::iterator data_begin,
                                                                 std::vector<uint32_t>::iterator data_end,
                                                                 size_t num_blocks, unsigned int num_threads) {
  size_t n = std::distance(data_begin, data_end);
  if (num_blocks <= 1 || n <= 1) {
    return;
  }
  std::vector<std::vector<uint32_t>::iterator> block_boundaries;
  block_boundaries.push_back(data_begin);
  size_t block_size = (n + num_blocks - 1) / num_blocks;
  for (size_t i = 1; i < num_blocks; ++i) {
    block_boundaries.push_back(data_begin + std::min(i * block_size, n));
  }
  block_boundaries.push_back(data_end);
  for (size_t pass = 0; pass < num_blocks; ++pass) {
    std::vector<std::function<void()>> merge_tasks;
    size_t start_block_idx = (pass % 2 == 0) ? 0 : 1;
    for (size_t i = start_block_idx; i + 1 < num_blocks; i += 2) {
      auto merge_begin = block_boundaries[i];
      auto merge_mid = block_boundaries[i + 1];
      auto merge_end = block_boundaries[i + 2];
      if (merge_mid >= merge_end) {
        continue;
      }
      merge_tasks.emplace_back(
          [merge_begin, merge_mid, merge_end]() { std::inplace_merge(merge_begin, merge_mid, merge_end); });
    }
    if (!merge_tasks.empty()) {
      ParallelRunTasks(merge_tasks, num_threads);
    }
  }
}