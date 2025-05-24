#include "all/opolin_d_radix_sort_batcher_merge/include/ops_all.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_invoke.h"

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + size_);
    unsigned int output_size = task_data->outputs_count[0];
    output_ = std::vector<int>(output_size, 0);
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::ValidationImpl() {
  if (world_.rank() == 0) {
    global_original_size_ = static_cast<int>(task_data->inputs_count[0]);
    if (global_original_size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
      return false;
    }
    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  std::vector<int> local_data;
  std::vector<int> counts(size), displs(size);
  int local_size = 0;
  if (rank == 0) {
    int chunk = size_ / size;
    int remainder = size_ % size;
    int offset = 0;
    for (int i = 0; i < size; ++i) {
      counts[i] = chunk + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += counts[i];
    }
  }

  boost::mpi::scatter(world_, counts, local_size, 0);
  local_data.resize(local_size);
  boost::mpi::scatterv(world_, input_.data(), counts, displs, local_data.data(), 0);

  std::vector<uint32_t> keys(local_size);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, local_size), [&](auto& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ConvertIntToUint(local_data[i]);
    }
  });

  RadixSort(keys);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, local_size), [&](auto& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      local_data[i] = ConvertUintToInt(keys[i]);
    }
  });

  BatcherOddEvenMerge(local_data, 0, local_size);
  std::vector<int> gathered_data;
  if (rank == 0) {
    gathered_data.resize(size_);
  }

  boost::mpi::gatherv(world_, local_data.data(), local_size, gathered_data.data(), counts, displs, 0);

  if (rank == 0) {
    output_.swap(gathered_data);
    for (int step = 1; step < size; step *= 2) {
      for (int left = 0; left < size; left += 2 * step) {
        int mid = std::min(left + step, size_);
        int right = std::min(left + 2 * step, size_);
        BatcherOddEvenMerge(output_, left, right);
      }
    }
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  } 
  return true;
}

uint32_t opolin_d_radix_batcher_sort_all::ConvertIntToUint(int num) { return static_cast<uint32_t>(num) ^ 0x80000000U; }

int opolin_d_radix_batcher_sort_all::ConvertUintToInt(uint32_t unum) { return static_cast<int>(unum ^ 0x80000000U); }

void opolin_d_radix_batcher_sort_all::RadixSort(std::vector<uint32_t>& uns_vec) {
  size_t sz = uns_vec.size();
  if (sz <= 1) {
    return;
  }
  const int rad = 256;
  std::vector<uint32_t> res(sz);
  for (int stage = 0; stage < 4; stage++) {
    tbb::enumerable_thread_specific<std::vector<size_t>> local_counts([&] { return std::vector<size_t>(rad, 0); });
    int shift = stage * 8;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, sz), [&](const tbb::blocked_range<size_t>& r) {
      auto& lc = local_counts.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        const uint8_t byte = (uns_vec[i] >> shift) & (rad - 1);
        lc[byte]++;
      }
    });
    std::vector<size_t> pref(rad, 0);
    for (auto& lc_instance : local_counts) {
      for (int j = 0; j < rad; ++j) {
        pref[j] += lc_instance[j];
      }
    }
    for (int j = 1; j < rad; ++j) {
      pref[j] += pref[j - 1];
    }
    for (int i = static_cast<int>(sz) - 1; i >= 0; --i) {
      const uint8_t byte = (uns_vec[i] >> shift) & (rad - 1);
      res[--pref[byte]] = uns_vec[i];
    }
    uns_vec.swap(res);
  }
}

void opolin_d_radix_batcher_sort_all::BatcherOddEvenMerge(std::vector<int>& vec, int low, int high) {
  if (high - low <= 1) return;
  int mid = (low + high) / 2;
  tbb::parallel_invoke([&] { BatcherOddEvenMerge(vec, low, mid); }, [&] { BatcherOddEvenMerge(vec, mid, high); });

  tbb::parallel_for(tbb::blocked_range<int>(low, mid), [&](const auto& r) {
    for (int i = r.begin(); i < r.end(); ++i)
      if (vec[i] > vec[i + mid - low]) {
        std::swap(vec[i], vec[i + mid - low]);
      }
  });
}