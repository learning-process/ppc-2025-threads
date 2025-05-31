#include "all/deryabin_m_hoare_sort_simple_merge/include/ops_all.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <algorithm>
#include <bit>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

void deryabin_m_hoare_sort_simple_merge_mpi::SeqHoaraSort(std::vector<double>::iterator first,
                                                          std::vector<double>::iterator last) {
  if (first >= last) {
    return;
  }
  if (last - first >= 199) {
    const auto mid = first + ((last - first) >> 1);
    const double pivot_value = [&]() {
      if (*first < *mid) {
        return *mid < *last ? *mid : std::max(*first, *last);
      }
      return *first < *last ? *first : std::max(*mid, *last);
    }();
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
  }
  if (last - first >= 199) {
    const auto mid = first + ((last - first) >> 1);
    const double pivot_value = [&]() {
      if (*first < *mid) {
        return *mid < *last ? *mid : std::max(*first, *last);
      }
      return *first < *last ? *first : std::max(*mid, *last);
    }();
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

void deryabin_m_hoare_sort_simple_merge_mpi::MergeUnequalTwoParts(std::vector<double>::iterator first,
                                                                  std::vector<double>::iterator mid,
                                                                  std::vector<double>::iterator last) {
  if (last - first >= 200) {
    const auto left_end = std::upper_bound(first, mid, *mid);
    const auto right_start = std::upper_bound(mid, last, *(mid - 1));
    const std::ptrdiff_t overlap_len = std::min(std::distance(left_end, mid), std::distance(mid, right_start));
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, static_cast<size_t>(overlap_len)),
                              [&left_end, &mid](const oneapi::tbb::blocked_range<size_t>& r) {
                                for (size_t i = r.begin(); i < r.end(); ++i) {
                                  auto left = left_end + static_cast<std::ptrdiff_t>(i);
                                  auto right = mid + static_cast<std::ptrdiff_t>(i);
                                  if (*left > *right) {
                                    std::iter_swap(left, right);
                                  }
                                }
                              });
    std::ptrdiff_t right_len = right_start - (mid - 1);
    std::ptrdiff_t left_len = mid - left_end;
    if (right_len > left_len + 1) {
      std::ptrdiff_t delta = right_len - left_len;
      auto* base = &*(right_start - delta);
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, static_cast<size_t>(left_len)),
                                [base, delta](const oneapi::tbb::blocked_range<size_t>& r) {
                                  for (size_t j = r.begin(); j < r.end(); ++j) {
                                    double* current = base - static_cast<std::ptrdiff_t>(j);
                                    for (std::ptrdiff_t i = 0; i < delta - 1; ++i) {
                                      if (current[i] > current[i + 1]) {
                                        std::swap(current[i], current[i + 1]);
                                      }
                                    }
                                  }
                                });
    }
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
    HoaraSort(input_array_A_.begin() + static_cast<std::ptrdiff_t>(count * min_chunk_size_),
              input_array_A_.begin() + static_cast<std::ptrdiff_t>((count + 1) * min_chunk_size_ - 1));
    count++;
  }
  size_t chunk_count = chunk_count_;
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_)) - 1; i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      std::inplace_merge(input_array_A_.begin() + static_cast<std::ptrdiff_t>(j * min_chunk_size_ << (i + 1)),
                         input_array_A_.begin() + static_cast<std::ptrdiff_t>((j << 1 | 1) * (min_chunk_size_ << i)),
                         input_array_A_.begin() + static_cast<std::ptrdiff_t>((j + 1) * min_chunk_size_ << (i + 1)));
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
    chunk_count_ = world_.size();
    min_chunk_size_ = dimension_ / chunk_count_;
    rest_ = dimension_ % chunk_count_;
  }
  boost::mpi::broadcast(world, dimension_, 0);
  if (world.rank() != 0) {
    input_array_A_.resize(dimension_);
  }
  boost::mpi::broadcast(world, input_array_A_.data(), static_cast<int>(dimension_), 0);
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
  const size_t rank_from_end = chunk_count_ - world_rank;
  const auto start_iter =
      input_array_A_.begin() +
      static_cast<std::ptrdiff_t>((rank_from_end - 1) * min_chunk_size_ + (world_rank == chunk_count_ - 1 ? 0 : rest_));
  const auto end_iter =
      input_array_A_.begin() + static_cast<std::ptrdiff_t>(rank_from_end * min_chunk_size_ + rest_ - 1);
  HoaraSort(start_iter, end_iter);
  const size_t iterations = std::bit_width(chunk_count_ - 1);
  for (size_t i = 0; i < iterations; ++i) {
    const size_t step = 1ULL << i;
    if ((chunk_count_ - world_rank) % step == 0 || world_rank == 0) {
      const bool is_even = (chunk_count_ & 1) != 0 ? ((chunk_count_ - world_rank + step - 1) / step & 1) != 0
                                                   : ((chunk_count_ - world_rank - 1) / step & 1) == 0;
      if (is_even) {
        if (world_rank == 0) {
          continue;
        }
        size_t block_size = min_chunk_size_ * step;
        size_t start_idx = (chunk_count_ - (world_rank + step)) * min_chunk_size_;
        if (world_rank == chunk_count_ - step) {
          block_size += rest_;
        } else {
          start_idx += rest_;
        }
        if (world_rank >= step) {
          world.send(static_cast<int>(world_rank - step), 0, input_array_A_.data() + start_idx,
                      static_cast<int>(block_size));
        } else {
          world.send(0, 0, input_array_A_.data() + start_idx, static_cast<int>(block_size));
        }
      } else {
        const bool special_odd_case = (chunk_count_ & 1) != 0 && world_rank == 0;
        size_t block_size = min_chunk_size_ * step;
        size_t start_idx = special_odd_case ? (chunk_count_ - (2 * step - 1)) * min_chunk_size_
                                            : (chunk_count_ - (world_rank + 2 * step)) * min_chunk_size_;
        const size_t recv_rank = special_odd_case ? step - 1 : world_rank + step;
        if (recv_rank == chunk_count_ - step) {
          block_size += rest_;
        } else {
          start_idx += rest_;
        }
        world.recv(static_cast<int>(recv_rank), 0, input_array_A_.data() + start_idx, static_cast<int>(block_size));
        const auto final_iter =
            special_odd_case ? input_array_A_.end()
                             : input_array_A_.begin() + static_cast<std::ptrdiff_t>(start_idx + block_size * 2 - rest_);
        MergeUnequalTwoParts(input_array_A_.begin() + static_cast<std::ptrdiff_t>(start_idx),
                             input_array_A_.begin() + static_cast<std::ptrdiff_t>(start_idx + block_size), final_iter);
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
