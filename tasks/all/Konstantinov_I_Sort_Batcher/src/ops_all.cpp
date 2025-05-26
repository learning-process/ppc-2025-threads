#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <ranges>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace konstantinov_i_sort_batcher_all {
namespace {

uint64_t ConvertDoubleForSorting(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(value));

  if ((bits >> 63) != 0) {
    return ~bits;
  }
  return bits ^ (1ULL << 63);
}

double RestoreDoubleFromSorted(uint64_t encoded_value) {
  if ((encoded_value >> 63) != 0) {
    encoded_value ^= (1ULL << 63);
  } else {
    encoded_value = ~encoded_value;
  }

  double result = 0.0;
  std::memcpy(&result, &encoded_value, sizeof(result));
  return result;
}

void ParallelRadixSort(std::vector<uint64_t>& data, int num_threads) {
  constexpr int kBitsPerPass = 8;
  constexpr int kTotalBits = 64;
  constexpr int kNumBucket = 256;

  std::vector<uint64_t> buffer(data.size());
  std::vector<std::vector<int>> thread_bucket_counts(num_threads, std::vector<int>(kNumBucket));

  for (int bit_shift = 0; bit_shift < kTotalBits; bit_shift += kBitsPerPass) {
    std::vector<std::thread> workers;
    const size_t elements_per_thread = (data.size() + num_threads - 1) / num_threads;

    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      const size_t start = thread_id * elements_per_thread;
      const size_t end = std::min(start + elements_per_thread, data.size());

      workers.emplace_back([&, start, end, thread_id]() {
        for (size_t i = start; i < end; ++i) {
          const auto bucket = static_cast<uint8_t>((data[i] >> bit_shift) & 0xFF);
          thread_bucket_counts[thread_id][bucket]++;
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    std::vector<int> global_counts(kNumBucket);
    for (int bucket = 0; bucket < kNumBucket; ++bucket) {
      for (int t = 0; t < num_threads; ++t) {
        global_counts[bucket] += thread_bucket_counts[t][bucket];
        thread_bucket_counts[t][bucket] = 0;
      }
    }

    for (int i = 1; i < kNumBucket; i++) {
      global_counts[i] += global_counts[i - 1];
    }

    for (int i = static_cast<int>(data.size()) - 1; i >= 0; i--) {
      const auto bucket = static_cast<uint8_t>((data[i] >> bit_shift) & 0xFF);
      buffer[--global_counts[bucket]] = data[i];
    }

    data.swap(buffer);
  }
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_all

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PreProcessingImpl() {
  const int rank = world_.rank();
  const int cluster_size = world_.size();

  if (rank == 0) {
    const size_t total_elements = task_data->inputs_count[0];
    const size_t per_node = (total_elements + cluster_size - 1) / cluster_size;
    mas_.resize(per_node * cluster_size, std::numeric_limits<double>::max());
    std::ranges::copy_n(reinterpret_cast<double*>(task_data->inputs[0]), total_elements, mas_.begin());
  }

  size_t elements_per_node = 0;
  if (rank == 0) {
    elements_per_node = mas_.size() / cluster_size;
  }
  boost::mpi::broadcast(world_, elements_per_node, 0);

  std::vector<double> local_data(elements_per_node);
  boost::mpi::scatter(world_, mas_, local_data.data(), elements_per_node, 0);
  mas_.swap(local_data);

  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }
  if (!task_data) {
    return false;
  }
  if (!task_data->inputs.data() || !task_data->inputs[0]) {
    return false;
  }
  if (!task_data->outputs.data() || !task_data->outputs[0]) {
    return false;
  }
  if (task_data->inputs_count[0] < 2) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::RunImpl() {
  const int rank = world_.rank();
  const int cluster_size = world_.size();

  std::vector<uint64_t> encoded_numbers;
  encoded_numbers.reserve(mas_.size());
  for (double value : mas_) {
    encoded_numbers.push_back(konstantinov_i_sort_batcher_all::ConvertDoubleForSorting(value));
  }

  const int optimal_threads =
      std::max(1, std::min(static_cast<int>(encoded_numbers.size()), ppc::util::GetPPCNumThreads()));
  konstantinov_i_sort_batcher_all::ParallelRadixSort(encoded_numbers, optimal_threads);

  const int merge_stages = static_cast<int>(std::ceil(std::log2(cluster_size)));
  for (int stage = 0; stage < merge_stages; ++stage) {
    int node_offset = 1 << (merge_stages - stage - 1);

    for (int merge_step = node_offset; merge_step > 0; merge_step >>= 1) {
      const int partner_node = rank ^ merge_step;
      if (partner_node >= cluster_size) {
        continue;
      }
      std::vector<uint64_t> partner_data(encoded_numbers.size());
      boost::mpi::request requests[2];

      if (rank < partner_node) {
        requests[0] = world_.isend(partner_node, 0, encoded_numbers.data(), static_cast<int>(encoded_numbers.size()));
        requests[1] = world_.irecv(partner_node, 0, partner_data.data(), static_cast<int>(partner_data.size()));
      } else {
        requests[0] = world_.irecv(partner_node, 0, partner_data.data(), static_cast<int>(partner_data.size()));
        requests[1] = world_.isend(partner_node, 0, encoded_numbers.data(), static_cast<int>(encoded_numbers.size()));
        boost::mpi::wait_all(requests, requests + 2);

        std::vector<uint64_t> merged_result;
        merged_result.reserve(encoded_numbers.size() * 2);
        std::ranges::merge(encoded_numbers, partner_data, std::back_inserter(merged_result));

        const auto keep_count = static_cast<std::ptrdiff_t>(encoded_numbers.size());

        if (rank < partner_node) {
          encoded_numbers.assign(merged_result.begin(), merged_result.begin() + keep_count);
        } else {
          encoded_numbers.assign(merged_result.end() - keep_count, merged_result.end());
        }
      }
      world_.barrier();
    }

    output_.resize(encoded_numbers.size());
    std::ranges::transform(encoded_numbers, output_.begin(), konstantinov_i_sort_batcher_all::RestoreDoubleFromSorted);
    return true;
  }
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PostProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();
  const int local_size = static_cast<int>(output_.size());

  std::vector<double> gathered;

  if (rank == 0) {
    gathered.resize(local_size * size);
  }

  boost::mpi::gather(world_, output_.data(), local_size, gathered.data(), 0);

  if (rank == 0) {
    auto removed = std::ranges::remove(gathered, std::numeric_limits<double>::max());
    gathered.erase(removed.begin(), removed.end());

    std::ranges::sort(gathered);

    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(gathered, out_ptr);
  }

  return true;
}