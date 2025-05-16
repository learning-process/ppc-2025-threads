#include "all/burykin_m_radix/include/ops_all.hpp"

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

  // Set up data distribution
  int local_size = input_size / world_.size();
  int remainder = input_size % world_.size();

  // Adjust local size for processes that get an extra element
  if (world_.rank() < remainder) {
    local_size += 1;
  }

  // Resize input vector for all processes
  input_.resize(local_size);

  // Distribute data from root to all processes
  if (world_.size() > 1) {
    std::vector<int> send_counts(world_.size());
    std::vector<int> displs(world_.size());

    for (int i = 0; i < world_.size(); ++i) {
      send_counts[i] = input_size / world_.size();
      if (i < remainder) {
        send_counts[i]++;
      }
      displs[i] = i > 0 ? displs[i - 1] + send_counts[i - 1] : 0;
    }

    // Use Scatterv with proper handling for root process
    if (world_.rank() == 0) {
      // Only copy data on root
      std::vector<int> temp_input(in_ptr, in_ptr + input_size);
      MPI_Scatterv(temp_input.data(), send_counts.data(), displs.data(), MPI_INT, input_.data(), local_size, MPI_INT, 0,
                   world_);
    } else {
      MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, input_.data(), local_size, MPI_INT, 0, world_);
    }
  } else {
    // Single process case - just copy the data
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }

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

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

  // Perform radix sort for each byte
  for (int shift = 0; shift < 32; shift += 8) {
    // Compute local frequency counts
    auto local_count = ComputeFrequency(a, shift);
    std::array<int, 256> global_count = {};

    // Sum up counts across all processes using all_reduce
    for (int i = 0; i < 256; ++i) {
      boost::mpi::all_reduce(world_, local_count[i], global_count[i], std::plus<int>());
    }

    // Calculate global indices based on global counts
    const auto global_index = ComputeIndices(global_count);

    // Calculate offset for current process
    std::array<int, 256> prefix_sum = {};

    // Calculate prefix sums for each bucket across processes
    for (int i = 0; i < 256; ++i) {
      std::vector<int> all_counts(world_.size());
      boost::mpi::all_gather(world_, local_count[i], all_counts);

      int sum = 0;
      for (int j = 0; j < world_.rank(); ++j) {
        sum += all_counts[j];
      }
      prefix_sum[i] = sum;
    }

    // Calculate local start indices
    std::array<int, 256> local_index = global_index;
    for (int i = 0; i < 256; ++i) {
      local_index[i] += prefix_sum[i];
    }

    // Distribute elements into buckets
    DistributeElements(a, b, local_index, shift);

    // Swap buffers for next iteration
    a.swap(b);
  }

  output_ = std::move(a);

  // Gather results to root process
  int total_size = task_data->inputs_count[0];
  std::vector<int> all_results;

  if (world_.rank() == 0) {
    all_results.resize(total_size);
  }

  // Calculate sizes and displacements for gathering
  std::vector<int> recv_counts(world_.size());
  std::vector<int> displs(world_.size());
  int remainder = total_size % world_.size();

  for (int i = 0; i < world_.size(); ++i) {
    recv_counts[i] = total_size / world_.size();
    if (i < remainder) {
      recv_counts[i]++;
    }
    displs[i] = i > 0 ? displs[i - 1] + recv_counts[i - 1] : 0;
  }

  // Gather results to root process
  MPI_Gatherv(output_.data(), output_.size(), MPI_INT, world_.rank() == 0 ? all_results.data() : nullptr,
              recv_counts.data(), displs.data(), MPI_INT, 0, world_);

  if (world_.rank() == 0) {
    output_ = all_results;
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