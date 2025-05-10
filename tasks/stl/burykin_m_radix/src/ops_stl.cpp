#include "stl/burykin_m_radix/include/ops_stl.hpp"

#include <array>
#include <cstddef>
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

void burykin_m_radix_stl::RadixSTL::ComputeFrequencyParallel(const std::vector<int>& a, const int shift, size_t start,
                                                             size_t end, std::array<int, 256>& local_count) {
  for (size_t i = start; i < end; ++i) {
    int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    ++local_count[key];
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

void burykin_m_radix_stl::RadixSTL::DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                                               const std::array<int, 256>& index, const int shift,
                                                               size_t start, size_t end,
                                                               std::vector<std::array<int, 256>>& local_indices) {
  std::array<int, 256> thread_index = index;
  size_t thread_id = local_indices.data() - &thread_index;

  for (size_t i = start; i < end; ++i) {
    int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    b[thread_index[key]++] = v;
    local_indices[thread_id][key] = thread_index[key];
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

  for (int shift = 0; shift < 32; shift += 8) {
    std::vector<std::array<int, 256>> thread_counts(num_threads, std::array<int, 256>{});
    std::vector<std::thread> threads(num_threads);

    const size_t chunk_size = a.size() / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? a.size() : (t + 1) * chunk_size;
      threads[t] = std::thread(ComputeFrequencyParallel, std::ref(a), shift, start, end, std::ref(thread_counts[t]));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    std::array<int, 256> count = {};
    for (const auto& local_count : thread_counts) {
      for (int i = 0; i < 256; ++i) {
        count[i] += local_count[i];
      }
    }

    const auto index = ComputeIndices(count);

    std::vector<std::array<int, 256>> thread_indices(num_threads);
    std::array<int, 256> current_index = index;

    for (int i = 0; i < 256; ++i) {
      int bucket_size = count[i];
      int per_thread = bucket_size / num_threads;
      int remainder = bucket_size % num_threads;

      int pos = current_index[i];
      for (int t = 0; t < num_threads; ++t) {
        thread_indices[t][i] = pos;
        int items = per_thread + (t < remainder ? 1 : 0);
        pos += items;
      }
    }

    for (int t = 0; t < num_threads; ++t) {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? a.size() : (t + 1) * chunk_size;
      threads[t] = std::thread(DistributeElementsParallel, std::ref(a), std::ref(b), std::ref(index), shift, start, end,
                               std::ref(thread_indices));
    }

    for (auto& thread : threads) {
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