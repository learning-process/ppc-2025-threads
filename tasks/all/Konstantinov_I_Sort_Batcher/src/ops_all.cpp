#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <algorithm>
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
void ParallelConvertToKeysMPI(std::vector<double>& arr, std::vector<uint64_t>& keys, boost::mpi::communicator& world) {
  size_t n = arr.size();
  int rank = world.rank();
  int size = world.size();

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  size_t base = n / size;
  size_t remainder = n % size;

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = static_cast<int>(base + (i < remainder ? 1 : 0));
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  std::vector<double> local_arr(sendcounts[rank]);

  // Используем Boost.MPI для передачи данных
  boost::mpi::scatterv(world, arr, sendcounts, displs, local_arr, 0);  // Используем корректные параметры

  std::vector<uint64_t> local_keys(local_arr.size());
  for (size_t i = 0; i < local_arr.size(); ++i) {
    local_keys[i] = DoubleToKey(local_arr[i]);
  }

  std::vector<int> recvcounts = sendcounts;
  std::vector<int> rdispls = displs;
  if (rank == 0) {
    keys.resize(n);
  }

  boost::mpi::gatherv(world, local_keys, keys, recvcounts, rdispls, 0);  // Используем правильную гаттер версию
}

void RadixPassMPI(std::vector<uint64_t>& arr, int shift, boost::mpi::communicator& world) {
  int rank = world.rank();
  int size = world.size();
  size_t n = arr.size();

  std::vector<int> sendcounts(size), displs(size);
  size_t base = n / size, rem = n % size;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = static_cast<int>(base + (i < rem ? 1 : 0));
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
  }

  std::vector<uint64_t> local_arr(sendcounts[rank]);
  boost::mpi::scatterv(world, arr, sendcounts, displs, local_arr, 0);

  const size_t radix = 256;
  std::vector<size_t> local_count(radix, 0);
  for (const auto& val : local_arr) {
    int byte = (val >> shift) & 0xFF;
    local_count[byte]++;
  }

  std::vector<size_t> global_count(radix, 0);
  boost::mpi::reduce(world, local_count.data(), local_count.data() + radix, global_count.data(), std::plus<size_t>(), 0);

  std::vector<uint64_t> sorted(n);
  if (rank == 0) {
    std::vector<size_t> offsets(radix, 0);
    for (size_t i = 1; i < radix; ++i) {
      offsets[i] = offsets[i - 1] + global_count[i - 1];
    }

    for (int proc = 0; proc < size; ++proc) {
      std::vector<uint64_t> temp;
      if (proc == 0) {
        temp = local_arr;
      } else {
        world.recv(proc, 1, temp);
      }

      for (const auto& val : temp) {
        int byte = (val >> shift) & 0xFF;
        sorted[offsets[byte]++] = val;
      }
    }

    arr = std::move(sorted);
  } else {
    world.send(0, 1, local_arr);
  }

  broadcast(world, arr, 0);
}

void ParallelConvertBackMPI(std::vector<uint64_t>& keys, std::vector<double>& arr, boost::mpi::communicator& world) {
  size_t n = keys.size();
  int rank = world.rank();
  int size = world.size();

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  size_t base = n / size;
  size_t rem = n % size;

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = base + (i < rem ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  std::vector<uint64_t> local_keys(sendcounts[rank]);

  boost::mpi::scatterv(world, keys, sendcounts, displs, local_keys, 0);

  std::vector<double> local_arr(local_keys.size());
  for (size_t i = 0; i < local_keys.size(); ++i) {
    local_arr[i] = KeyToDouble(local_keys[i]);
  }

  std::vector<int> recvcounts = sendcounts;
  std::vector<int> rdispls = displs;
  if (rank == 0) {
    arr.resize(n);
  }

  boost::mpi::gatherv(world, local_arr, arr, recvcounts, rdispls, 0);
}

void RadixSortMPI(std::vector<double>& arr, boost::mpi::communicator& world) {
  int rank = world.rank();
  int size = world.size();
  size_t n = arr.size();
  std::vector<uint64_t> keys(n);

  ParallelConvertToKeysMPI(arr, keys, world);

  std::vector<uint64_t> output_keys(n);

  for (int pass = 0; pass < 8; ++pass) {
    RadixPassMPI(keys, pass, world);
    keys.swap(output_keys);
  }

  ParallelConvertBackMPI(keys, arr, world);
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
  int rank = world.rank();

  RadixSortMPI(arr, world);

  if (rank == 0) {
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