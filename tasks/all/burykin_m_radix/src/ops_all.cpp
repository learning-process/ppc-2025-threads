#include "all/burykin_m_radix/include/ops_all.hpp"

#include <omp.h>

#include <array>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeFrequency(const std::vector<int>& a, const int shift,
                                                                     int start, int end) {
  std::array<int, 256> count = {};
  for (int i = start; i < end; ++i) {
    const int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    ++count[key];
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
                                                       std::array<int, 256> index, const int shift, int start,
                                                       int end) {
  for (int i = start; i < end; ++i) {
    const int v = a[i];
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }

#pragma omp atomic capture
    {
      b[index[key]] = v;
      index[key]++;
    }
  }
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  if (world_.rank() == 0) {
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }

  boost::mpi::broadcast(world_, input_, 0);
  output_.resize(input_.size());

  return true;
}

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

  for (int shift = 0; shift < 32; shift += 8) {
    std::array<int, 256> local_count = {};
    std::array<int, 256> global_count = {};

    const int size = static_cast<int>(a.size());
    const int rank = world_.rank();
    const int num_procs = world_.size();

    const int chunk_size = size / num_procs;
    const int start = rank * chunk_size;
    const int end = (rank == num_procs - 1) ? size : (rank + 1) * chunk_size;

#pragma omp parallel default(none) shared(a, b, shift, local_count, start, end)
    {
      std::array<int, 256> thread_count = {};

#pragma omp for
      for (int i = start; i < end; ++i) {
        const int v = a[i];
        unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }
        ++thread_count[key];
      }

#pragma omp critical
      {
        for (int i = 0; i < 256; ++i) {
          local_count[i] += thread_count[i];
        }
      }
    }

    boost::mpi::all_reduce(world_, local_count.data(), 256, global_count.data(), std::plus<int>());

    const auto index = ComputeIndices(global_count);
    std::array<int, 256> working_index = index;

    world_.barrier();

#pragma omp parallel default(none) shared(a, b, shift, working_index)
    {
#pragma omp for
      for (size_t i = 0; i < a.size(); ++i) {
        const int v = a[i];
        unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }

        int pos;
#pragma omp atomic capture
        { pos = working_index[key]++; }
        b[pos] = v;
      }
    }

    world_.barrier();
    a.swap(b);
  }

  output_ = std::move(a);

  if (world_.rank() != 0) {
    boost::mpi::gather(world_, output_.data(), output_.size(), 0);
  } else {
    std::vector<int> gathered_result;
    boost::mpi::gather(world_, output_.data(), output_.size(), gathered_result, 0);
    if (!gathered_result.empty()) {
      output_ = std::move(gathered_result);
    }
  }

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}