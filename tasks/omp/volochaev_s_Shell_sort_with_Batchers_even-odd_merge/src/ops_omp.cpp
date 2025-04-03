#include "omp/example/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "omp/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_omp.hpp"

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PreProcessingImpl() {
  // Init value for input and output
  size_ = static_cast<int>(task_data->inputs_count[0]);

  auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
  array_ = std::vector<int>(input_pointer, input_pointer + size_);
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

// void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ShellSort() {
//   int n = finish - start;
//   int gap = n / 2;
//
//   while (gap > 0) {
//     for (int i = start + gap; i < finish; ++i) {
//       int temp = p_data[i];
//       int j = i;
//       while (j >= gap && p_data[j - gap] > temp) {
//         p_data[j] = p_data[j - gap];
//         j -= gap;
//       }
//       p_data[j] = temp;
//     }
//     gap /= 2;
//     gap /= 2;
//   }
// }

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ShellSort(int start) {
  int n = mini_batch_;

  int gap = 1;
  while (gap < n / 3) {
    gap = 3 * gap + 1;
  }

  while (gap >= 1) {
    for (int i = start + gap; i < start + mini_batch_; ++i) {
      int temp = mass_[i];
      int j = i;
      while (j >= start + gap && mass_[j - gap] > temp) {
        mass_[j] = mass_[j - gap];
        j -= gap;
      }
      mass_[j] = temp;
    }
    gap /= 3;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::MergeLast(int start, int id1, int id2,
                                                                                      int len, int c) {
  while (id1 < len) {
    array_[start + id2] = mass_[start + id1];
    id1 += 2;
    id2 += c;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::MergeBlocks(int id_l, int id_r, int len) {
  int left_id = 0;
  int right_id = 0;
  int merged_id = 0;

  while (left_id < len || right_id < len) {
    if (left_id < len && right_id < len) {
      if (mass_[id_l + left_id] < mass_[id_r + right_id]) {
        array_[id_l + merged_id] = mass_[id_l + left_id];
        left_id += 2;
      } else {
        array_[id_l + merged_id] = mass_[id_r + right_id];
        right_id += 2;
      }
    } else if (left_id < len) {
      array_[id_l + merged_id] = mass_[id_l + left_id];
      left_id += 2;
    } else {
      array_[id_l + merged_id] = mass_[id_r + right_id];
      right_id += 2;
    }
    merged_id += 2;
  }

  for (int i = 0; i < merged_id; i += 2) {
    mass_[id_l + i] = array_[id_l + i];
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::LastMerge() {
  int even_index = 0;
  int odd_index = 1;
  int result_index = 0;

  while (even_index < n_ || odd_index < n_) {
    if (even_index < n_ && odd_index < n_) {
      if (mass_[even_index] < mass_[odd_index]) {
        array_[result_index++] = mass_[even_index];
        even_index += 2;
      } else {
        array_[result_index++] = mass_[odd_index];
        odd_index += 2;
      }
    } else if (even_index < n_) {
      array_[result_index++] = mass_[even_index];
      even_index += 2;
    } else {
      array_[result_index++] = mass_[odd_index];
      odd_index += 2;
    }
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::Merge() {
  for (int i = 0; i < n_; ++i) {
    std::cout << mass_[i] << " ";
  }

  std::cout << '\n';
  for (int i = c_threads_; i > 1; i /= 2) {
#pragma omp parallel num_threads(i)
    {
      int id = omp_get_thread_num() / 2;
      int ost = omp_get_thread_num() % 2;
      int l = mini_batch_ * (c_threads_ / i);

      MergeBlocks((id * 2 * l) + ost, (id * 2 * l) + l + ost, l - ost);
    }

    for (int k = 0; k < n_; ++k) {
      std::cout << mass_[k] << " ";
    }
    std::cout << '\n';
  }

  LastMerge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ParallelShellSort() {
  FindThreadVariables();
  std::vector<int> index(c_threads_);

  for (int i = 0; i < c_threads_; i++) {
    index[i] = i * mini_batch_;
  }

#pragma omp parallel for
  for (int i = 0; i < c_threads_; ++i) {
    ShellSort(i * mini_batch_);
  }

  Merge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::FindThreadVariables() {
  c_threads_ = static_cast<int>(std::pow(2, std::floor(std::log2(omp_get_max_threads()))));
  n_ = size_ + (((2 * c_threads_) - size_ % (2 * c_threads_))) % (2 * c_threads_);
  std::cout << "n_: " << n_ << '\n';
  mass_.resize(n_);
  mini_batch_ = n_ / c_threads_;
  std::cout << "mini_batch_: " << mini_batch_ << '\n';
  for (int i = 0; i < size_; ++i) {
    mass_[i] = array_[i];
  }

  for (int i = size_; i < static_cast<int>(mass_.size()); ++i) {
    mass_[i] = std::numeric_limits<int>::max();
  }
  array_.resize(n_);
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::RunImpl() {
  ParallelShellSort();
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PostProcessingImpl() {
  for (int i = 0; i < size_; i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = array_[i];
  }
  return true;
}