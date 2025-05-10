#include "stl/burykin_m_radix/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

std::array<int, 256> burykin_m_radix_stl::RadixSTL::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};
  for (const int v : a) {
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    ++count[key];
  }
  return count;
}

void ComputeFrequencyParallel(const std::vector<int>& a, const int shift, std::array<int, 256>& count, int start_idx,
                              int end_idx) {
  std::array<int, 256> local_count = {};
  for (int i = start_idx; i < end_idx; ++i) {
    const int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    ++local_count[key];
  }

  for (int i = 0; i < 256; ++i) {
    count[i] += local_count[i];
  }
}

std::array<int, 256> burykin_m_radix_stl::RadixSTL::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_stl::RadixSTL::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  for (const int v : a) {
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    b[index[key]++] = v;
  }
}

void DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                const std::array<int, 256>& global_index,
                                std::vector<std::array<int, 256>>& local_counts, const int shift, int thread_id,
                                int start_idx, int end_idx) {
  std::array<int, 256> index = global_index;

  for (int i = 0; i < 256; ++i) {
    for (int t = 0; t < thread_id; ++t) {
      index[i] += local_counts[t][i];
    }
  }

  for (int i = start_idx; i < end_idx; ++i) {
    const int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    b[index[key]++] = v;
  }
}

bool burykin_m_radix_stl::RadixSTL::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_stl::RadixSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_stl::RadixSTL::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

  const int num_threads = ppc::util::GetPPCNumThreads();
  const size_t elements_per_thread = a.size() / num_threads + ((a.size() % num_threads) ? 1 : 0);

  for (int shift = 0; shift < 32; shift += 8) {
    std::array<int, 256> count = {};
    std::vector<std::thread> freq_threads;
    std::vector<std::array<int, 256>> local_counts(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      int start_idx = t * elements_per_thread;
      int end_idx = std::min(start_idx + elements_per_thread, a.size());

      if (start_idx < a.size()) {
        freq_threads.emplace_back([&a, shift, &local_counts, t, start_idx, end_idx]() {
          ComputeFrequencyParallel(a, shift, local_counts[t], start_idx, end_idx);
        });
      }
    }

    for (auto& thread : freq_threads) {
      thread.join();
    }

    for (const auto& local_count : local_counts) {
      for (int i = 0; i < 256; ++i) {
        count[i] += local_count[i];
      }
    }

    const auto index = ComputeIndices(count);

    std::vector<std::thread> dist_threads;

    std::vector<std::array<int, 256>> thread_key_counts(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      int start_idx = t * elements_per_thread;
      int end_idx = std::min(start_idx + elements_per_thread, a.size());

      if (start_idx < a.size()) {
        for (int i = start_idx; i < end_idx; ++i) {
          const int v = a[i];
          unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
          if (shift == 24) {
            key ^= 0x80;
          }
          ++thread_key_counts[t][key];
        }
      }
    }

    for (int t = 0; t < num_threads; ++t) {
      int start_idx = t * elements_per_thread;
      int end_idx = std::min(start_idx + elements_per_thread, a.size());

      if (start_idx < a.size()) {
        dist_threads.emplace_back([&a, &b, &index, &thread_key_counts, shift, t, start_idx, end_idx]() {
          DistributeElementsParallel(a, b, index, thread_key_counts, shift, t, start_idx, end_idx);
        });
      }
    }

    for (auto& thread : dist_threads) {
      thread.join();
    }

    a.swap(b);
  }

  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_stl::RadixSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}