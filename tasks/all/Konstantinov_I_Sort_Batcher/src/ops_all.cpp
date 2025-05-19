#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"

#include <algorithm>
#include <atomic>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace konstantinov_i_sort_batcher_all {
namespace {
uint64_t DoubleToKey(double d) {
  uint64_t u = 0;
  std::memcpy(&u, &d, sizeof(d));

  if ((u >> 63) != 0) {
    return ~u;
  }
  return u ^ 0x8000000000000000ULL;
}

double KeyToDouble(uint64_t key) {
  if ((key >> 63) != 0) {
    key = key ^ 0x8000000000000000ULL;
  } else {
    key = ~key;
  }
  double d = NAN;
  std::memcpy(&d, &key, sizeof(d));
  return d;
}

void ParallelConvertToKeys(std::vector<double>& arr, std::vector<uint64_t>& keys, int thread_count) {
  size_t n = arr.size();
  size_t block_size = (n + thread_count - 1) / thread_count;
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&arr, &keys, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        keys[i] = DoubleToKey(arr[i]);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}
void RadixPass(std::vector<uint64_t>& keys, std::vector<uint64_t>& output_keys, int pass, int thread_count) {
  size_t n = keys.size();
  size_t block_size = (n + thread_count - 1) / thread_count;
  int shift = pass * 8;
  const int radix = 256;

  std::vector<std::vector<size_t>> local_counts(thread_count, std::vector<size_t>(radix, 0));
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([t, &keys, &local_counts, shift, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
        local_counts[t][byte]++;
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  std::vector<size_t> count(radix, 0);
  for (int b = 0; b < radix; ++b) {
    for (int t = 0; t < thread_count; ++t) {
      count[b] += local_counts[t][b];
    }
  }
  for (int i = 1; i < radix; ++i) {
    count[i] += count[i - 1];
  }
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    auto byte = static_cast<uint8_t>((keys[i] >> shift) & 0xFF);
    output_keys[--count[byte]] = keys[i];
  }
}
void ParallelConvertBack(std::vector<uint64_t>& keys, std::vector<double>& arr, int thread_count) {
  size_t n = keys.size();
  size_t block_size = (n + thread_count - 1) / thread_count;
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&arr, &keys, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        arr[i] = KeyToDouble(keys[i]);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

void RadixSorted(std::vector<double>& arr) {
  if (arr.empty()) {
    return;
  }
  size_t n = arr.size();
  std::vector<uint64_t> keys(n);
  const int thread_count = ppc::util::GetPPCNumThreads();

  ParallelConvertToKeys(arr, keys, thread_count);

  std::vector<uint64_t> output_keys(n);
  for (int pass = 0; pass < 8; ++pass) {
    RadixPass(keys, output_keys, pass, thread_count);
    keys.swap(output_keys);
  }

  ParallelConvertBack(keys, arr, thread_count);
}

void BatcherOddEvenMerge(std::vector<double>& arr, int low, int high) {
  if (high - low <= 1) {
    return;
  }
  int mid = (low + high) / 2;

  BatcherOddEvenMerge(arr, low, mid);
  BatcherOddEvenMerge(arr, mid, high);

  for (int i = low; i < mid; ++i) {
    if (arr[i] > arr[i + mid - low]) {
      std::swap(arr[i], arr[i + mid - low]);
    }
  }
}

void ParallelRadixSortBoostMPI(std::vector<double>& local_arr, boost::mpi::communicator& comm) {
  int rank = comm.rank();
  int size = comm.size();

  RadixSorted(local_arr);

  std::vector<std::vector<double>> all_data;
  boost::mpi::all_gather(comm, local_arr, all_data);

  std::vector<double> combined_data;
  combined_data.reserve(size * local_arr.size());

  for (const auto& vec : all_data) {
    combined_data.insert(combined_data.end(), vec.begin(), vec.end());
  }

  BatcherOddEvenMerge(combined_data, 0, static_cast<int>(combined_data.size()));

  size_t start = 0;
  for (int i = 0; i < rank; ++i) {
    start += all_data[i].size();
  }
  local_arr.assign(combined_data.begin() + start, combined_data.begin() + start + local_arr.size());
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_all

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::ValidationImpl() {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    bool is_valid = (task_data->inputs_count[0] == task_data->outputs_count[0]);

    boost::mpi::broadcast(world, is_valid, 0);
    return is_valid;
  } else {
    bool is_valid;
    boost::mpi::broadcast(world, is_valid, 0);
    return is_valid;
  }
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PreProcessingImpl() {
  boost::mpi::communicator world;

  if (!ValidationImpl()) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  unsigned int local_size = input_size / world.size();
  unsigned int start = world.rank() * local_size;
  if (world.rank() == world.size() - 1) {
    local_size = input_size - start;
  }

  mas_ = std::vector<double>(in_ptr + start, in_ptr + start + local_size);
  output_ = std::vector<double>(task_data->outputs_count[0], 0);

  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::RunImpl() {
  boost::mpi::communicator world;
  if (!ValidationImpl()) return false;

  output_ = mas_;
  ParallelRadixSortBoostMPI(output_, world);
  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PostProcessingImpl() {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<std::vector<double>> gathered_data;
    boost::mpi::gather(world, output_, gathered_data, 0);

    std::vector<double> combined;
    for (const auto& vec : gathered_data) {
      combined.insert(combined.end(), vec.begin(), vec.end());
    }

    std::copy(combined.begin(), combined.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  } else {
    boost::mpi::gather(world, output_, 0);
  }

  boost::mpi::broadcast(world, reinterpret_cast<double*>(task_data->outputs[0]), task_data->outputs_count[0], 0);

  return true;
}