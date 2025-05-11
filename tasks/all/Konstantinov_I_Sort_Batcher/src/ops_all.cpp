#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

void RadixPass(std::vector<uint64_t>& arr, int shift) {
  const size_t radix = 256;
  std::vector<size_t> count(radix, 0);
  for (auto& num : arr) count[(num >> (shift * 8)) & 0xFF]++;

  std::vector<size_t> offset(radix, 0);
  for (size_t i = 1; i < radix; ++i) offset[i] = offset[i - 1] + count[i - 1];

  std::vector<uint64_t> output(arr.size());
  for (auto& num : arr) {
    int idx = (num >> (shift * 8)) & 0xFF;
    output[offset[idx]++] = num;
  }
  arr.swap(output);
}

void RadixSortMPI(std::vector<double>& arr, boost::mpi::communicator& world) {
  int rank = world.rank();
  int size = world.size();
  size_t n = arr.size();

  std::vector<int> sendcounts(size), displs(size);
  int base = n / size, rem = n % size;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0 ? 0 : displs[i - 1] + sendcounts[i - 1]);
  }

  std::vector<double> local_arr(sendcounts[rank]);

  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      std::vector<double> temp(arr.begin() + displs[i], arr.begin() + displs[i] + sendcounts[i]);
      world.send(i, 0, temp);
    }
    std::copy(arr.begin(), arr.begin() + sendcounts[0], local_arr.begin());
  } else {
    world.recv(0, 0, local_arr);
  }

  std::vector<uint64_t> local_keys(local_arr.size());
  for (size_t i = 0; i < local_arr.size(); ++i) local_keys[i] = DoubleToKey(local_arr[i]);

  for (int pass = 0; pass < 8; ++pass) RadixPass(local_keys, pass);

  for (size_t i = 0; i < local_arr.size(); ++i) local_arr[i] = KeyToDouble(local_keys[i]);

  if (rank == 0) {
    std::copy(local_arr.begin(), local_arr.end(), arr.begin());
    for (int i = 1; i < size; ++i) {
      std::vector<double> temp(sendcounts[i]);
      world.recv(i, 1, temp);
      std::copy(temp.begin(), temp.end(), arr.begin() + displs[i]);
    }
  } else {
    world.send(0, 1, local_arr);
  }
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
  boost::mpi::communicator world;
  RadixSortMPI(arr, world);

  if (world.rank() == 0) {
    BatcherOddEvenMerge(arr, 0, static_cast<int>(arr.size()));
  }
}  // namespace
}  // namespace konstantinov_i_sort_batcher_all

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  mas_ = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::RunImpl() {
  output_ = mas_;
  konstantinov_i_sort_batcher_all::RadixSort(output_);
  return true;
}

bool konstantinov_i_sort_batcher_all::RadixSortBatcherall::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}