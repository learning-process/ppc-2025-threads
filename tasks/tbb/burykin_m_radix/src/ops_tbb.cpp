#include "tbb/burykin_m_radix/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <array>
#include <utility>
#include <vector>

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeFrequency(const std::vector<int>& a, const int shift) {
  // Using enumerable_thread_specific for thread-local counters
  tbb::enumerable_thread_specific<std::array<int, 256>> local_counts(std::array<int, 256>{});

  // Parallel counting of frequencies
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a.size()), [&](const tbb::blocked_range<size_t>& range) {
    auto& my_count = local_counts.local();
    for (size_t i = range.begin(); i < range.end(); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++my_count[key];
    }
  });

  // Combine all thread-local counters
  std::array<int, 256> count = {};
  for (const auto& local_count : local_counts) {
    for (int i = 0; i < 256; ++i) {
      count[i] += local_count[i];
    }
  }

  return count;
}

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  // Sequential dependency, cannot be easily parallelized
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_tbb::RadixTBB::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  // Create per-thread offsets for each bucket
  struct BucketOffsets {
    std::array<int, 256> offsets;
    BucketOffsets() : offsets() {}
  };

  tbb::enumerable_thread_specific<BucketOffsets> local_offsets;

  // First pass: count elements per thread and bucket
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a.size()), [&](const tbb::blocked_range<size_t>& range) {
    auto& my_offsets = local_offsets.local();
    for (size_t i = range.begin(); i < range.end(); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++my_offsets.offsets[key];
    }
  });

  // Calculate starting positions for each thread and bucket
  std::vector<std::array<int, 256>> prefix_sums;
  prefix_sums.reserve(local_offsets.size() + 1);

  // Start with the original indices
  prefix_sums.push_back(index);

  // Add each thread's count
  for (const auto& offset : local_offsets) {
    std::array<int, 256> next_sum = prefix_sums.back();
    for (int i = 0; i < 256; ++i) {
      next_sum[i] += offset.offsets[i];
    }
    prefix_sums.push_back(next_sum);
  }

  // Reset thread-local buckets for the second pass
  local_offsets.clear();

  // Second pass: distribute elements to output array
  tbb::parallel_for(tbb::blocked_range<size_t>(0, a.size()), [&](const tbb::blocked_range<size_t>& range) {
    // Get thread-specific starting positions
    int my_id = 0;
    {
      static tbb::enumerable_thread_specific<int> tls_id(0);
      auto& local_id = tls_id.local();
      if (local_id == 0) {
        static std::atomic<int> next_id(0);
        local_id = next_id++;
      }
      my_id = local_id;
    }

    std::array<int, 256> my_index = prefix_sums[my_id];

    for (size_t i = range.begin(); i < range.end(); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      b[my_index[key]++] = v;
    }
  });
}

bool burykin_m_radix_tbb::RadixTBB::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_tbb::RadixTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_tbb::RadixTBB::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

  tbb::task_arena arena;
  arena.execute([&] {
    for (int shift = 0; shift < 32; shift += 8) {
      auto count = ComputeFrequency(a, shift);
      const auto index = ComputeIndices(count);
      DistributeElements(a, b, index, shift);
      a.swap(b);
    }
  });

  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_tbb::RadixTBB::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, output_.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i < range.end(); ++i) {
      output_ptr[i] = output_[i];
    }
  });

  return true;
}