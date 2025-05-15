#include "all/burykin_m_radix/include/ops_all.hpp"

#include <omp.h>

#include <array>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

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

  if (world_.rank() == 0) {
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  } else {
    input_.resize(input_size);
  }

  boost::mpi::broadcast(world_, input_, 0);

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

  const int rank = world_.rank();
  const int world_size = world_.size();

  int chunk_size = static_cast<int>(input_.size()) / world_size;
  int start_idx = rank * chunk_size;
  int end_idx = (rank == world_size - 1) ? static_cast<int>(input_.size()) : (rank + 1) * chunk_size;

  std::vector<int> local_input(input_.begin() + start_idx, input_.begin() + end_idx);
  std::vector<int> local_output(local_input.size());

  std::vector<int> a = std::move(local_input);
  std::vector<int> b(a.size());

#pragma omp parallel
  {
#pragma omp single
    {
      for (int shift = 0; shift < 32; shift += 8) {
        auto local_count = ComputeFrequency(a, shift);

        std::array<int, 256> global_count = {};
        boost::mpi::all_reduce(world_, local_count.data(), 256, global_count.data(), std::plus<int>());

        auto global_index = ComputeIndices(global_count);

        std::array<int, 256> process_offsets = {};
        for (int i = 0; i < 256; ++i) {
          int count_before_this_process = 0;
          for (int p = 0; p < rank; ++p) {
            int process_count = 0;
            boost::mpi::broadcast(world_, process_count, p);
            count_before_this_process += process_count;
          }
          process_offsets[i] = global_index[i] + count_before_this_process;
        }

        DistributeElements(a, b, process_offsets, shift);

        std::vector<int> all_data;
        boost::mpi::all_gather(world_, b.data(), b.size(), all_data);

        a = std::move(all_data);
        b.resize(a.size());

        world_.barrier();
      }
    }
  }

  local_output = std::move(a);

  boost::mpi::gather(world_, local_output.data(), local_output.size(), output_, 0);

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    const auto output_size = static_cast<int>(output_.size());

#pragma omp parallel for
    for (int i = 0; i < output_size; ++i) {
      output_ptr[i] = output_[i];
    }
  }
  return true;
}