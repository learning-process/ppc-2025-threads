#include "tbb/burykin_m_radix/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeFrequencyParallel(const std::vector<int>& a,
                                                                             const int shift) {
  std::array<std::array<int, 256>, 64> local_counts = {};
  const size_t array_size = a.size();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, array_size), [&](const tbb::blocked_range<size_t>& range) {
    std::array<int, 256> local_count = {};
    for (size_t i = range.begin(); i < range.end(); ++i) {
      unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++local_count[key];
    }

    // Store local histogram in the local_counts array
    int thread_id = tbb::this_task_arena::current_thread_index();
    thread_id = (thread_id < 0) ? 0 : thread_id % 64;  // Ensure thread index is within bounds
    local_counts[thread_id] = local_count;
  });

  // Combine all local histograms
  std::array<int, 256> global_count = {};
  for (const auto& count : local_counts) {
    for (int i = 0; i < 256; ++i) {
      global_count[i] += count[i];
    }
  }

  return global_count;
}

std::array<int, 256> burykin_m_radix_tbb::RadixTBB::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_tbb::RadixTBB::DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                                               const std::array<int, 256>& index, const int shift) {
  // Calculate offsets for each key using standard atomic
  std::array<std::atomic<int>, 256> offsets;
  for (int i = 0; i < 256; ++i) {
    offsets[i] = index[i];
  }

  tbb::parallel_for(tbb::blocked_range<size_t>(0, a.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i < range.end(); ++i) {
      unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      // Use standard atomic fetch_add instead of fetch_and_increment
      int idx = offsets[key].fetch_add(1, std::memory_order_relaxed);
      b[idx] = a[i];
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

  // Configure TBB task arena with the number of threads from PPC
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());

  arena.execute([&] {
    std::vector<int> a = std::move(input_);
    std::vector<int> b(a.size());

    for (int shift = 0; shift < 32; shift += 8) {
      auto count = ComputeFrequencyParallel(a, shift);
      const auto index = ComputeIndices(count);
      DistributeElementsParallel(a, b, index, shift);
      a.swap(b);
    }

    output_ = std::move(a);
  });

  return true;
}

bool burykin_m_radix_tbb::RadixTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}