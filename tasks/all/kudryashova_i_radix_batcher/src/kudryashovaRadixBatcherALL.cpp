#include "all/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherALL.hpp"

#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/partitioner.h>

#include <algorithm>
#include <array>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mpi = boost::mpi;

void kudryashova_i_radix_batcher_all::ConvertDoublesToUint64(const std::vector<double>& data,
                                                             std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = 0;
          memcpy(&bits, &data[first + i], sizeof(double));
          converted[i] = ((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63);
        }
      },
      tbb::auto_partitioner());
}

void kudryashova_i_radix_batcher_all::ConvertUint64ToDoubles(std::vector<double>& data,
                                                             const std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = converted[i];
          bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
          memcpy(&data[first + i], &bits, sizeof(double));
        }
      },
      tbb::auto_partitioner());
}

void kudryashova_i_radix_batcher_all::RadixDoubleSort(std::vector<double>& data, size_t first, size_t last) {
  const size_t sort_size = last - first;
  std::vector<uint64_t> converted(sort_size);
  ConvertDoublesToUint64(data, converted, first);

  std::vector<uint64_t> buffer(sort_size);
  int bits_int_byte = 8;
  int max_byte_value = 255;
  size_t total_bits = sizeof(uint64_t) * CHAR_BIT;
  for (size_t shift = 0; shift < total_bits; shift += bits_int_byte) {
    tbb::combinable<std::array<size_t, 256>> local_counts;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sort_size), [&](const auto& range) {
      auto& counts = local_counts.local();
      for (size_t i = range.begin(); i != range.end(); ++i) {
        ++counts[(converted[i] >> shift) & max_byte_value];
      }
    });

    std::array<size_t, 256> total_counts{};
    local_counts.combine_each([&](const auto& local_count) {
      for (size_t i = 0; i < 256; ++i) {
        total_counts[i] += local_count[i];
      }
    });
    size_t total = 0;
    for (auto& safe : total_counts) {
      size_t old = safe;
      safe = total;
      total += old;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, 256), [&](const auto& range) {
      for (size_t j = range.begin(); j != range.end(); ++j) {
        size_t count = total_counts[j];
        for (size_t i = 0; i < sort_size; ++i) {
          if (((converted[i] >> shift) & max_byte_value) == j) {
            buffer[count++] = converted[i];
          }
        }
      }
    });

    converted.swap(buffer);
  }
  ConvertUint64ToDoubles(data, converted, first);
}

void kudryashova_i_radix_batcher_all::BatcherMerge(std::vector<double>& target_array, size_t merge_start,
                                                   size_t mid_point, size_t merge_end) {
  const size_t total_elements = merge_end - merge_start;
  const size_t left_size = mid_point - merge_start;
  const size_t right_size = merge_end - mid_point;
  std::vector<double> left_array(target_array.begin() + static_cast<std::vector<double>::difference_type>(merge_start),
                                 target_array.begin() + static_cast<std::vector<double>::difference_type>(mid_point));
  std::vector<double> right_array(target_array.begin() + static_cast<std::vector<double>::difference_type>(mid_point),
                                  target_array.begin() + static_cast<std::vector<double>::difference_type>(merge_end));
  size_t left_ptr = 0;
  size_t right_ptr = 0;
  size_t merge_ptr = merge_start;
  for (size_t i = 0; i < total_elements; ++i) {
    if (i % 2 == 0) {
      if (left_ptr < left_size && (right_ptr >= right_size || left_array[left_ptr] <= right_array[right_ptr])) {
        target_array[merge_ptr++] = left_array[left_ptr++];
      } else {
        target_array[merge_ptr++] = right_array[right_ptr++];
      }
    } else {
      if (right_ptr < right_size && (left_ptr >= left_size || right_array[right_ptr] <= left_array[left_ptr])) {
        target_array[merge_ptr++] = right_array[right_ptr++];
      } else {
        target_array[merge_ptr++] = left_array[left_ptr++];
      }
    }
  }
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
      return false;
    }
    input_data_.resize(task_data->inputs_count[0]);
    std::copy_n(reinterpret_cast<double*>(task_data->inputs[0]), task_data->inputs_count[0], input_data_.begin());

    n_ = input_data_.size();
    int base_chunk = n_ / world_.size();
    int remainder = n_ % world_.size();

    counts_.resize(world_.size());
    displs_.resize(world_.size());

    for (int i = 0; i < world_.size(); ++i) {
      int chunk_size = (i < remainder) ? base_chunk + 1 : base_chunk;
      counts_[i] = static_cast<int>(chunk_size);
      displs_[i] = (i == 0) ? 0 : displs_[i - 1] + counts_[i - 1];
    }
  }
  mpi::broadcast(world_, n_, 0);
  mpi::broadcast(world_, counts_, 0);
  mpi::broadcast(world_, displs_, 0);

  return true;
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::RunImpl() {
  int local_size = 0;
  mpi::scatter(world_, counts_, local_size, 0);
  std::vector<double> local_data(local_size);
  boost::mpi::scatterv(world_, world_.rank() == 0 ? input_data_.data() : nullptr, counts_, displs_, local_data.data(),
                       local_size, 0);
  RadixDoubleSort(local_data, 0, local_data.size());
  std::vector<double> global_data;
  if (world_.rank() == 0) {
    global_data.resize(n_);
  }
  boost::mpi::gatherv(world_, local_data.data(), static_cast<int>(local_data.size()), global_data.data(), counts_,
                      displs_, 0);
  if (world_.rank() == 0) {
    input_data_ = global_data;
    size_t current_merge_size = n_ / world_.size();
    if (n_ % world_.size() != 0) {
      current_merge_size = (n_ + world_.size() - 1) / world_.size();
    }

    for (; current_merge_size < n_; current_merge_size *= 2) {
      size_t merge_group_size = 2 * current_merge_size;
      size_t total_merge_groups = (n_ + merge_group_size - 1) / merge_group_size;

      tbb::parallel_for(tbb::blocked_range<size_t>(0, total_merge_groups), [&](const auto& range) {
        for (size_t group_index = range.begin(); group_index != range.end(); ++group_index) {
          size_t merge_start = group_index * merge_group_size;
          size_t merge_mid = std::min(merge_start + current_merge_size, n_);
          size_t merge_end = std::min(merge_start + merge_group_size, n_);
          if (merge_mid < merge_end) {
            BatcherMerge(input_data_, merge_start, merge_mid, merge_end);
          }
        }
      });
    }
  }

  size_t total_size = task_data->inputs_count[0];
  std::vector<int> local_sizes(world_.size());
  mpi::gather(world_, static_cast<int>(input_data_.size()), local_sizes, 0);

  if (world_.rank() == 0) {
    all_data_.resize(total_size);
    displs_[0] = 0;
    for (int p = 1; p < world_.size(); ++p) {
      displs_[p] = displs_[p - 1] + local_sizes[p - 1];
    }
  }

  boost::mpi::gatherv(world_, input_data_.data(), static_cast<int>(input_data_.size()), all_data_.data(), local_sizes,
                      displs_, 0);
  return true;
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return !input_data_.empty() && task_data->outputs_count[0] == input_data_.size() &&
           task_data->outputs[0] != nullptr;
  }
  return true;
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(all_data_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
