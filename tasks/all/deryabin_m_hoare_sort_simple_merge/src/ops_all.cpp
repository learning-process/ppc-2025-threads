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

void deryabin_m_hoare_sort_simple_merge_mpi::SeqHoaraSort(std::vector<double>::iterator first,
                                                          std::vector<double>::iterator last) {
  if (first >= last) {
    return;
  } else if (last - first >= 199) {
    const auto mid = first + ((last - first) >> 1);
    const double pivot_value = *first < *mid    ? *mid < *last ? *mid : std::max(*first, *last)
                                  : *first < *last ? *first
                                                : std::max(*mid, *last);
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
  } else {
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
}

void deryabin_m_hoare_sort_simple_merge_mpi::HoaraSort(std::vector<double>::iterator first,
                                                       std::vector<double>::iterator last) {
  if (first >= last) {
    return;
  } else if (last - first >= 199) {
    const auto mid = first + ((last - first) >> 1);
    const double pivot_value = *first < *mid    ? *mid < *last ? *mid : std::max(*first, *last)
                                  : *first < *last ? *first
                                                : std::max(*mid, *last);
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
    oneapi::tbb::parallel_invoke([&first, &right]() { HoaraSort(first, right); },
                                 [&left, &last]() { HoaraSort(left + 1, last); });
  } else {
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
}

void deryabin_m_hoare_sort_simple_merge_mpi::MergeTwoParts(std::vector<double>::iterator first,
                                                           std::vector<double>::iterator last) {
  if (last - first >= 200) {
    const auto mid = first + ((last - first) >> 1);
    const auto left_end = std::upper_bound(first, mid, *mid);
    const auto right_start = std::upper_bound(mid, last, *(mid - 1));
    const size_t overlap_len = std::min(std::distance(left_end, mid), std::distance(mid, right_start));
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, overlap_len),
                              [&left_end, &mid](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i = r.begin(); i < r.end(); ++i) {
                                  auto left = left_end + i;
                                  auto right = mid + i;
                                  if (*left > *right) {
                                    std::iter_swap(left, right);
                                  }
                                }
                              });
    size_t right_len = right_start - (mid - 1);
    size_t left_len = mid - left_end;
    if (right_len > left_len + 1) {
      size_t delta = right_len - left_len;
      auto base = &*(right_start - delta);
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, left_len),
                                [&base, delta](const oneapi::tbb::blocked_range<size_t>& r) {
                                  for (size_t j = r.begin(); j < r.end(); ++j) {
                                    double* current = base - j;
                                    for (size_t i = 0; i < delta - 1; ++i) {
                                      if (current[i] > current[i + 1]) {
                                        std::swap(current[i], current[i + 1]);
                                      }
                                    }
                                  }
                                });
    }
    std::inplace_merge(left_end, mid, right_start);
  } else {
    std::inplace_merge(first, first + ((last - first) >> 1), last);
  }
}

void deryabin_m_hoare_sort_simple_merge_mpi::MergeUnequalTwoParts(std::vector<double>::iterator first,
                                                                  std::vector<double>::iterator mid,
                                                                  std::vector<double>::iterator last) {
  if (last - first >= 200) {
    const auto left_end = std::upper_bound(first, mid, *mid);
    const auto right_start = std::upper_bound(mid, last, *(mid - 1));
    const size_t overlap_len = std::min(std::distance(left_end, mid), std::distance(mid - 1, right_start));
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, overlap_len),
                              [&left_end, &mid](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i = r.begin(); i < r.end(); ++i) {
                                  auto left = left_end + i;
                                  auto right = mid + i;
                                  if (*left > *right) {
                                    std::iter_swap(left, right);
                                  }
                                }
                              });
    std::inplace_merge(left_end, mid, right_start);
  } else {
    std::inplace_merge(first, mid, last);
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
  return task_data->inputs_count[0] > 2 && task_data->inputs_count[1] >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_.begin() + count * min_chunk_size_,
              input_array_A_.begin() + (count + 1) * min_chunk_size_ - 1);
    count++;
  }
  size_t chunk_count = chunk_count_;
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_)) - 1; i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      std::inplace_merge(input_array_A_.begin() + (j * min_chunk_size_ << (i + 1)),
                         input_array_A_.begin() + ((j << 1 | 1) * (min_chunk_size_ << i)),
                         input_array_A_.begin() + ((j + 1) * min_chunk_size_ << (i + 1)));
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
    chunk_count_ = world.size();
    min_chunk_size_ = dimension_ / chunk_count_;
    rest_ = dimension_ % chunk_count_;
  }
  boost::mpi::broadcast(world, dimension_, 0);
  if (world.rank() != 0) {
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
    return task_data->inputs_count[0] > 2 && task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::RunImpl() {
  const size_t world_rank = world.rank();
  const bool is_last_rank = world_rank == chunk_count_ - 1;
  const size_t rank_from_end = chunk_count_ - world_rank;
  const auto start_iter = input_array_A_.begin() + (rank_from_end - 1) * min_chunk_size_ + (!is_last_rank ? rest_ : 0);
  const auto end_iter = input_array_A_.begin() + rank_from_end * min_chunk_size_ + rest_ - 1;
  HoaraSort(start_iter, end_iter);
  const size_t iterations = std::bit_width(chunk_count_ - 1);
  for (size_t i = 0; i < iterations; ++i) {
    const size_t step = 1ULL << i;
    size_t block_size = min_chunk_size_ * step;
    if ((chunk_count_ - world_rank) % step == 0 || world_rank == 0) {
      const bool is_even = chunk_count_ & 1 ? ((chunk_count_ - world_rank + step - 1) / step & 1) != 0
                                            : ((chunk_count_ - world_rank - 1) / step & 1) == 0;
      if (is_even) {
        if (world_rank == 0) continue;
        size_t start_idx = (chunk_count_ - (world_rank + step)) * min_chunk_size_;
        if (!is_last_rank) {
          start_idx += rest_;
        } else {
          block_size += rest_;
        }
        if (world_rank >= step) {
          world.send(world_rank - step, 0, input_array_A_.data() + start_idx, block_size);
        } else {
          world.send(0, 0, input_array_A_.data() + start_idx, block_size);
        }
      } else {
        size_t start_idx = (chunk_count_ & 1) && world_rank == 0
                               ? (chunk_count_ - (2 * step - 1)) * min_chunk_size_
                               : (chunk_count_ - (world_rank + 2 * step)) * min_chunk_size_;
        const size_t recv_rank = (chunk_count_ & 1) && world_rank == 0 ? step - 1 : world_rank + step;
        if (recv_rank != chunk_count_ - 1) {
          start_idx += rest_;
        } else {
          block_size += rest_;
        }
        world.recv(recv_rank, 0, input_array_A_.data() + start_idx, block_size);
        if ((chunk_count_ & 1) && world_rank == 0) {
          const size_t merge_point = static_cast<size_t>((2.0 - 1.0 / step) * block_size);
          MergeUnequalTwoParts(input_array_A_.begin() + start_idx, input_array_A_.begin() + start_idx + block_size,
                               input_array_A_.begin() + start_idx + merge_point);
        } else {
          MergeTwoParts(input_array_A_.begin() + start_idx, input_array_A_.begin() + start_idx + block_size * 2);
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
