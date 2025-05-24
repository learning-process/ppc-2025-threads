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

void RadixSort(std::vector<double>& arr) {
  RadixSorted(arr);
  BatcherOddEvenMerge(arr, 0, static_cast<int>(arr.size()));
}
}  // namespace
}  // namespace konstantinov_i_sort_batcher_all

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PreProcessingImpl() {
  world_ = mpi::communicator();
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    mas_ = std::vector<double>(in_ptr, in_ptr + input_size);

    unsigned int output_size = task_data->outputs_count[0];
    output_ = std::vector<double>(output_size, 0);
  }
  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::ValidationImpl() {
  if (world_.rank() == 0) {
    bool valid = task_data->inputs_count[0] == task_data->outputs_count[0];
    broadcast(world_, valid, 0);
    return valid;
  } else {
    bool valid;
    broadcast(world_, valid, 0);
    return valid;
  }
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::RunImpl() {
  if (world_.size() == 1) {
    if (world_.rank() == 0) {
      output_ = mas_;
      konstantinov_i_sort_batcher_all::RadixSort(output_);
    }
    return true;
  }
  std::vector<double> local_data;
  if (world_.rank() == 0) {
    size_t total_size = mas_.size();
    size_t chunk_size = total_size / world_.size();
    size_t remainder = total_size % world_.size();

    for (int proc = 0; proc < world_.size(); ++proc) {
      size_t start = proc * chunk_size + std::min(proc, (int)remainder);
      size_t end = start + chunk_size + (proc < remainder ? 1 : 0);

      std::vector<double> chunk(mas_.begin() + start, mas_.begin() + end);
      if (proc == 0) {
        local_data = chunk;
      } else {
        world_.send(proc, 0, chunk);
      }
    }
  } else {
    world_.recv(0, 0, local_data);
  }
  konstantinov_i_sort_batcher_all::RadixSort(local_data);
  if (world_.rank() == 0) {
    output_.clear();
    output_.reserve(mas_.size());
    output_.insert(output_.end(), local_data.begin(), local_data.end());

    for (int proc = 1; proc < world_.size(); ++proc) {
      std::vector<double> proc_data;
      world_.recv(proc, 0, proc_data);
      output_.insert(output_.end(), proc_data.begin(), proc_data.end());
    }
    konstantinov_i_sort_batcher_all::BatcherOddEvenMerge(output_, 0, static_cast<int>(output_.size()));
  } else {
    world_.send(0, 0, local_data);
  }

  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}