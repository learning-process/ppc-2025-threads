#include "all/burykin_m_radix/include/ops_all.hpp"

#include <array>
#include <utility>
#include <vector>

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};

#pragma omp parallel default(none) shared(a, count, shift)
  {
    std::array<int, 256> local_count = {};

#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
      const int v = a[i];
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      ++local_count[key];
    }

#pragma omp critical
    {
      for (int i = 0; i < 256; ++i) {
        count[i] += local_count[i];
      }
    }
  }

  return count;
}

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_all::RadixALL::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  std::array<int, 256> local_index = index;
  std::vector<int> offsets(a.size());

#pragma omp parallel for default(none) shared(a, offsets, local_index, shift)
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    const int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }

    int pos = 0;
#pragma omp critical
    {
      pos = local_index[key];
      local_index[key]++;
    }

    offsets[i] = pos;
  }

#pragma omp parallel for default(none) shared(a, b, offsets)
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    b[offsets[i]] = a[i];
  }
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = input_;
  std::vector<int> b(a.size());

  if (world_.rank() == 0) {
#pragma omp parallel
    {
#pragma omp single
      {
        for (int shift = 0; shift < 32; shift += 8) {
          auto count = ComputeFrequency(a, shift);
          const auto index = ComputeIndices(count);
          DistributeElements(a, b, index, shift);
          a.swap(b);
        }
      }
    }
  } else {
#pragma omp parallel
    {
#pragma omp single
      {
        volatile int size = 0;
        for (int shift = 0; shift < 32; shift += 8) {
          int local_shift = size + shift;
          auto count = ComputeFrequency(a, shift);
          const auto index = ComputeIndices(count);
          DistributeElements(a, b, index, shift);
          a.swap(b);
        }
      }
    }
  }

  world_.barrier();
  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  const auto output_size = static_cast<int>(output_.size());

#pragma omp parallel for
  for (int i = 0; i < output_size; ++i) {
    output_ptr[i] = output_[i];
  }
  return true;
}