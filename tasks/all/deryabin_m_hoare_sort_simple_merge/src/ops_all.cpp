#include "all/deryabin_m_hoare_sort_simple_merge/include/ops_all.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <algorithm>
#include <bit>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

void deryabin_m_hoare_sort_simple_merge_mpi::HoaraSort(std::vector<double>::iterator first,
                                                       std::vector<double>::iterator last) {
  if (first >= last) {
    return;
  }
  const double pivot_value = *(first + ((last - first) >> 1));
  auto left = first;
  auto right = last;
  do {
    while (left < last && *left < pivot_value) {
      left++;
    }
    while (right > first && *right > pivot_value) {
      right--;
    }
    if (*left == *right && left != right) {
      if (*left < *(left + 1)) {
        left++;
      } else {
        right--;
      }
    }
    std::iter_swap(left, right);
  } while (left != right);
  if (last - first >= 199) {
    oneapi::tbb::parallel_invoke([&first, &right]() { HoaraSort(first, right); },
                                 [&left, &last]() { HoaraSort(left + 1, last); });
  } else {
    HoaraSort(first, right);
    HoaraSort(left + 1, last);
  }
}

void deryabin_m_hoare_sort_simple_merge_mpi::MergeTwoParts(std::vector<double>::iterator first,
                                                           std::vector<double>::iterator last) {
  const size_t len = std::distance(first, last);
  if (len <= 1) return;
  std::vector<double>::iterator mid = first + len / 2;
  std::vector<double>::iterator left_end = std::upper_bound(first, mid, *mid);
  std::vector<double>::iterator right_start = std::lower_bound(mid, last, *(mid - 1));
  const size_t overlap_len = std::distance(left_end, mid) + std::distance(mid, right_start);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, overlap_len), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::vector<double>::iterator left = left_end + i;
      std::vector<double>::iterator right = mid + i;
      if (*left > *right) {
        std::iter_swap(left, right);
      }
    }
  });
  std::inplace_merge(left_end, mid, right_start);
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_.begin() + count * min_chunk_size_,
              input_array_A_.begin() + ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  size_t chunk_count = chunk_count_;
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) - 1); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      std::inplace_merge(input_array_A_.begin() + static_cast<long>(j * min_chunk_size_ << (i + 1)),
                         input_array_A_.begin() + static_cast<long>(((j << 1 | 1) * (min_chunk_size_ << i))),
                         input_array_A_.begin() + static_cast<long>((j + 1) * min_chunk_size_ << (i + 1)));
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
    dimension_ = task_data->inputs_count[0];
    chunk_count_ = static_cast<size_t>(world.size());
    min_chunk_size_ = dimension_ / chunk_count_;
    rest_ = dimension_ % chunk_count_;
  }
  boost::mpi::broadcast(world, dimension_, 0);
  if (world.rank() != 0) {
    input_array_A_.reserve(dimension_);
    input_array_A_.resize(dimension_);
  }
  boost::mpi::broadcast(world, input_array_A_.data(), dimension_, 0);
  boost::mpi::broadcast(world, chunk_count_, 0);
  boost::mpi::broadcast(world, min_chunk_size_, 0);
  boost::mpi::broadcast(world, rest_, 0);
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::ValidationImpl() {
  if (world.rank() == 0) {
    return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
           task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::RunImpl() {
  auto start_iter = input_array_A_.begin() + (chunk_count_ - 1) * min_chunk_size_;
  if (world.rank() != 0) {
    start_iter += rest_;
  }
  HoaraSort(start_iter, input_array_A_.begin() + chunk_count_ * min_chunk_size_ + rest_ - 1);
  if (world.size() != 1) {
    for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_ - 1)); ++i) {
      unsigned short step = 1ULL << i;
      size_t block_size = min_chunk_size_ * static_cast<size_t>(step);
      if ((world.rank() + 1) % step == 0) {
        if (world.rank() / step % 2 == 0) {
          size_t start_idx =
              (static_cast<size_t>(world.rank() - step) + 1) * min_chunk_size_ + world.rank() != 0 ? rest_ : 0;
          world.send(static_cast<size_t>(world.rank() + step), 0, input_array_A_.data() + start_idx, block_size);
        }
        if (world.rank() / step % 2 != 0 || world.rank() == world.size() - 1) {
          size_t start_idx =
              (static_cast<size_t>(world.rank() - 2 * step) + 1) * min_chunk_size_ + world.rank() - step != 0 ? rest_
                                                                                                              : 0;
          world.recv(static_cast<size_t>(world.rank() - step), 0, input_array_A_.data() + start_idx, block_size);
          MergeTwoParts(input_array_A_.begin() + start_idx, input_array_A_.begin() + start_idx + 2 * block_size);
        }
      }
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  }
  return true;
}
