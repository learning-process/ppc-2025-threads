#include "omp/example/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
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

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ShellSort(int start, int finish) {
  int n = mini_batch_;
  int gap = n / 2;

  while (gap > 0) {
    for (int i = start + gap; i < finish; ++i) {
      int temp = mass_[i];
      int j = i;
      while (j >= gap && mass_[j - gap] > temp) {
        mass_[j] = mass_[j - gap];
        j -= gap;
      }
      mass_[j] = temp;
    }
    gap /= 2;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::MergeBlocks(int id_arr, int id_l, int id_r,
                                                                                        int len_l, int len_r) {
  int runner0 = 0;
  int runner1 = 0;
  int runnerarray = 0;

  while (runner0 < len_l && runner1 < len_r) {
    if (mass_[id_l + runner0] < mass_[id_r + runner1]) {
      array_[id_arr + runnerarray] = mass_[id_l + runner0];
      runner0 += 2;
    } else {
      array_[id_arr + runnerarray] = mass_[id_r + runner1];
      runner1 += 2;
    }

    runnerarray += 2;
  }

  while (runner0 < len_l) {
    array_[id_arr + runnerarray] = mass_[id_l + runner0];
    runner0 += 2;
    runnerarray += 2;
  }

  while (runner1 < len_r) {
    array_[id_arr + runnerarray] = mass_[id_r + runner1];
    runner1 += 2;
    runnerarray += 2;
  }

  for (int i = 0; i < runnerarray; i += 2) {
    mass_[id_l + i] = array_[id_arr + i];
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::LastMerge() {
  int runner0 = 0;
  int runner1 = 1;
  int runnerarray = 0;

  while (runner0 < n_ && runner1 < n_) {
    if (mass_[runner0] < mass_[runner1]) {
      array_[runnerarray] = mass_[runner0];
      runner0 += 2;
    } else {
      array_[runnerarray] = mass_[runner1];
      runner1 += 2;
    }

    runnerarray++;
  }

  while (runner0 < n_) {
    array_[runnerarray] = mass_[runner0];
    runner0 += 2;
    runnerarray++;
  }

  while (runner1 < n_) {
    array_[runnerarray] = mass_[runner1];
    runner1 += 2;
    runnerarray++;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::Merge() {
  for (int i = c_threads_; i > 1; i /= 2) {
#pragma omp parallel num_threads(i)
    {
      int id = static_cast<int>(static_cast<double>(omp_get_thread_num()) / 2);
      int ost = omp_get_thread_num() % 2;
      int l = mini_batch_ * (c_threads_ / i);

      MergeBlocks((id * 2 * l) + ost, (id * 2 * l) + ost, (id * 2 * l) + l + ost, l - ost, l - ost);
    }
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
    for (int i = 0; i < c_threads_; i++) {
      ShellSort(index[i], index[i] + mini_batch_);
    }

  Merge();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::FindThreadVariables() {
  c_threads_ = static_cast<int>(std::pow(2, std::floor(std::log2(omp_get_max_threads()))));
  n_ = size_ + (((2 * c_threads_) - size_ % (2 * c_threads_))) % (2 * c_threads_);
  mass_.resize(n_);
  mini_batch_ = n_ / c_threads_;
  int max_elem = std::numeric_limits<int>::min();
  for (int i = 0; i < size_; ++i) {
    mass_[i] = array_[i];
    max_elem = std::max(max_elem, array_[i]);
  }

  for (int i = size_; i < static_cast<int>(mass_.size()); ++i) {
    n_ = size_;
    mass_[i] = max_elem;
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