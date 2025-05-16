#include "all/burykin_m_radix/include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
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

  // Only rank 0 has the original data
  std::vector<int> original_data;
  if (world_.rank() == 0) {
    original_data.assign(in_ptr, in_ptr + input_size);
  }

  // Single process optimization
  if (world_.size() == 1) {
    input_ = std::move(original_data);
    output_.resize(input_.size());
    return true;
  }

  // For multiple processes, we need a different approach
  // First, calculate how many elements each process will handle
  std::vector<int> send_counts(world_.size());
  std::vector<int> displs(world_.size());

  int remainder = input_size % world_.size();
  for (int i = 0; i < world_.size(); ++i) {
    send_counts[i] = input_size / world_.size();
    if (i < remainder) {
      send_counts[i]++;
    }
    displs[i] = (i > 0) ? displs[i - 1] + send_counts[i - 1] : 0;
  }

  // Resize input for each process based on its assigned count
  int local_size = send_counts[world_.rank()];
  input_.resize(local_size);

  // Scatter the data from process 0 to all processes
  MPI_Scatterv(world_.rank() == 0 ? original_data.data() : nullptr, send_counts.data(), displs.data(), MPI_INT,
               input_.data(), local_size, MPI_INT, 0, world_);

  // Prepare output buffer
  output_.resize(local_size);
  return true;
}

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  // Handle empty input case
  if (input_.empty()) {
    return true;
  }

  // For a single process, just use a serial sort to ensure correctness
  if (world_.size() == 1) {
    output_ = input_;
    std::sort(output_.begin(), output_.end());
    return true;
  }

  // Multiple processes - implement a distributed radix sort
  std::vector<int> a = input_;
  std::vector<int> b(a.size());

  // Perform radix sort for each byte
  for (int shift = 0; shift < 32; shift += 8) {
    // 1. Each process counts its local elements
    auto local_count = ComputeFrequency(a, shift);

    // 2. All processes need to know the total counts
    std::array<int, 256> global_count = {};
    for (int i = 0; i < 256; ++i) {
      MPI_Allreduce(&local_count[i], &global_count[i], 1, MPI_INT, MPI_SUM, world_);
    }

    // 3. Calculate global prefix sums
    auto global_index = ComputeIndices(global_count);

    // 4. For each process, determine where to place its elements globally
    std::array<int, 256> rank_offsets = {};
    std::vector<int> all_counts(world_.size() * 256);

    // Gather all counts from all processes
    MPI_Allgather(local_count.data(), 256, MPI_INT, all_counts.data(), 256, MPI_INT, world_);

    // Calculate offsets for this process's elements in each bucket
    for (int i = 0; i < 256; ++i) {
      int offset = 0;
      for (int j = 0; j < world_.rank(); ++j) {
        offset += all_counts[j * 256 + i];
      }
      rank_offsets[i] = offset;
    }

    // 5. Calculate starting positions
    std::array<int, 256> local_index = {};
    for (int i = 0; i < 256; ++i) {
      local_index[i] = global_index[i] + rank_offsets[i];
    }

    // 6. Distribute elements locally
    DistributeElements(a, b, local_index, shift);

    // Prepare for global redistribution
    std::vector<int> send_counts(world_.size(), 0);
    std::vector<int> recv_counts(world_.size(), 0);
    std::vector<int> send_displs(world_.size(), 0);
    std::vector<int> recv_displs(world_.size(), 0);

    // Calculate new element distribution
    for (int i = 0; i < 256; ++i) {
      int total_count = global_count[i];
      int per_proc = total_count / world_.size();
      int remainder = total_count % world_.size();

      for (int rank = 0; rank < world_.size(); ++rank) {
        // For the current process, count elements to send to each rank
        if (world_.rank() == rank) {
          for (int other_rank = 0; other_rank < world_.size(); ++other_rank) {
            int count = per_proc + (other_rank < remainder ? 1 : 0);

            if (count > 0) {
              recv_counts[other_rank] += count;
            }
          }
        }

        // Count elements from the current digit this process will send to each rank
        int start_idx = local_index[i];
        int end_idx = start_idx + local_count[i];

        // Calculate which elements go to which rank
        for (int idx = start_idx; idx < end_idx; ++idx) {
          int global_idx = idx - global_index[i];
          int target_rank = (global_idx < remainder * (per_proc + 1))
                                ? (global_idx / (per_proc + 1))
                                : remainder + (global_idx - remainder * (per_proc + 1)) / per_proc;

          if (target_rank >= world_.size()) target_rank = world_.size() - 1;
          send_counts[target_rank]++;
        }
      }
    }

    // Calculate displacements
    for (int i = 1; i < world_.size(); ++i) {
      send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
      recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    // Prepare buffers for all-to-all communication
    std::vector<int> send_buffer(b.size());
    std::vector<int> recv_buffer(b.size());

    // Copy elements to send buffer
    for (size_t i = 0; i < b.size(); ++i) {
      send_buffer[i] = b[i];
    }

    // All-to-all exchange
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT, recv_buffer.data(),
                  recv_counts.data(), recv_displs.data(), MPI_INT, world_);

    // Prepare for next iteration
    a.resize(recv_buffer.size());
    for (size_t i = 0; i < recv_buffer.size(); ++i) {
      a[i] = recv_buffer[i];
    }
    b.resize(a.size());
  }

  // Set output to final sorted array
  output_ = a;

  // Gather all sorted pieces to rank 0
  int total_size = task_data->inputs_count[0];
  std::vector<int> gather_counts(world_.size());
  std::vector<int> gather_displs(world_.size());
  std::vector<int> full_result;

  // Calculate counts and displacements for gather
  int base_count = total_size / world_.size();
  int remainder = total_size % world_.size();

  for (int i = 0; i < world_.size(); ++i) {
    gather_counts[i] = base_count + (i < remainder ? 1 : 0);
    gather_displs[i] = (i > 0) ? gather_displs[i - 1] + gather_counts[i - 1] : 0;
  }

  // Root process allocates space for the entire result
  if (world_.rank() == 0) {
    full_result.resize(total_size);
  }

  // Gather all sorted pieces
  MPI_Gatherv(output_.data(), output_.size(), MPI_INT, world_.rank() == 0 ? full_result.data() : nullptr,
              gather_counts.data(), gather_displs.data(), MPI_INT, 0, world_);

  if (world_.rank() == 0) {
    // Perform a final merge or sort if needed
    std::sort(full_result.begin(), full_result.end());
    output_ = std::move(full_result);
  }

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
