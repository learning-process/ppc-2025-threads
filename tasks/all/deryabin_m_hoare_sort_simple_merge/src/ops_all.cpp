#include "all/deryabin_m_hoare_sort_simple_merge/include/ops_all.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <algorithm>
#include <bit>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

void deryabin_m_hoare_sort_simple_merge_mpi::SeqHoaraSort(std::vector<double>::iterator first,
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
  HoaraSort(first, right);
  HoaraSort(left + 1, last);
}

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
  if (last - first >= 200) {
    const size_t len = last - first;
    if (len <= 1) return;
    const auto mid = first + (len >> 1);
    const auto pivot = *mid;
    const auto left_end = std::upper_bound(first, mid, pivot);
    const auto right_pivot = (mid > first) ? *(mid - 1) : *mid;
    const auto right_start = std::lower_bound(mid, last, right_pivot);
    const size_t left_len = mid - left_end;
    const size_t right_len = right_start - mid;
    const size_t overlap_len = left_len + right_len;
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, overlap_len),
                              [&left_end, &mid](const oneapi::tbb::blocked_range<size_t>& r) {
                                auto left = left_end + r.begin();
                                auto right = mid + r.begin();
                                const auto end = left + (r.end() - r.begin());
                                while (left != end) {
                                  if (*left > *right) {
                                    std::iter_swap(left, right);
                                  } else if (*left == *right) {
                                    ++left;
                                    ++right;
                                    continue;
                                  }
                                  ++left;
                                  ++right;
                                }
                              });
    std::inplace_merge(left_end, mid, right_start);
  } else {
    std::inplace_merge(first, first + ((last - first) >> 1), last);
  }
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
  if (world.rank() == 0) {
    unsigned short k = 1;
    while (k != world.size()) {
      world.send(k, 0, input_array_A_.data(), dimension_);
      k++;
    }
  } else {
    input_array_A_.resize(dimension_);
    world.recv(0, 0, input_array_A_.data(), dimension_);
  }
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
  const auto chunk_size = min_chunk_size_;
  auto start_iter = input_array_A_.begin() + static_cast<size_t>(world.size() - world.rank() - 1) * chunk_size;
  if (world.rank() != world.size() - 1) {
    start_iter += rest_;
  }
  const auto end_iter =
      input_array_A_.begin() + static_cast<size_t>(world.size() - world.rank()) * chunk_size + rest_ - 1;
  HoaraSort(start_iter, end_iter);
  const auto world_size = world.size();
  const size_t iterations = static_cast<size_t>(std::bit_width(chunk_count_ - 1));
  for (size_t i = 0; i < iterations; ++i) {
    const unsigned short step = 1ULL << i;
    const size_t block_size = chunk_size * step;
    if ((world_size - world.rank()) % step == 0 || world.rank() == 0) {
      const bool is_even = ((world_size - world.rank() - 1) / step % 2 == 0);
      if (world_size % 2 != 0) {
        is_even = ((world_size - world.rank() + step - 1) / step % 2 != 0);
      }
      if (is_even) {
        size_t start_idx = static_cast<size_t>(world_size - (world.rank() + step)) * chunk_size;
        if (world.rank() != world.size() - 1) {
          start_idx += rest_;
        }
        if (world.rank() - step >= 0) {
          world.send(world.rank() - step, 0, &input_array_A_[start_idx], block_size);
        } else {
          world.send(0, 0, &input_array_A_[start_idx], block_size);
        }
      } else {
        size_tstart_idx = static_cast<size_t>(world_size - (world.rank() + 2 * step)) * chunk_size;
        if (world.rank() + step != world.size() - 1) {
          start_idx += rest_;
        }
        if (world_size % 2 != 0 && world.rank() == 0) {
          world.recv(world.rank() + step - 1, 0, &input_array_A_[start_idx], block_size);
          MergeTwoParts(input_array_A_.begin() + start_idx, input_array_A_.begin() + start_idx + (1 + world.rank() / step) * block_size);
        } else {
          world.recv(world.rank() + step, 0, &input_array_A_[start_idx], block_size);
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
