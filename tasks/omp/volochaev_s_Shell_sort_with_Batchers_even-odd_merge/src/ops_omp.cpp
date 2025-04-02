#include "omp/example/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "omp/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_omp.hpp"

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PreProcessingImpl() {
  // Init value for input and output
  size_ = std::static_cast<int>(task_data->inputs_count[0]);

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
  int n = finish - start;
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
    gap /= 2;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ParallelShellSort() {
  FindThreadVariables();
  std::vector<int> index(c_threads_ + 1);
  std::vector<int> block_size(c_threads_, mini_batch_);
  std::vector<int> block_ids(c_threads_);

  for (int i = 0; i <= c_threads_; i++) {
    index[i] = (i * n) / c_threads_;
    block_ids[i] = i;
  }

#pragma omp parallel for
  for (int i = 0; i < c_threads_; i++) {
    ShellSort(index[i], index[i] + block_size[i]);
  }

  for (int i = 1; i < c_threads_; i *= 2) {
#pragma omp parallel for
    for (int j = 0; j < c_threads_; j += 2 * i) {
      int left = block_ids[j];
      int right = -1;
      if (j + i < c_threads_) {
        right = block_ids[j + i];
      }

      if (right != -1) {
        MergeBlocks(index[left], block_size[left], index[right], block_size[right]);
        block_ids[j] = left;
        block_size[j] = block_size[left] + block_size[right];
      } else {
        block_ids[j] = left;
        block_size[j] = block_size[left];
      }
    }
  }

  for (int i = 0; i < size_; ++i) {
    input_[i] = mass_[i];
  }
}

void MergeBlocks(int id1, int sz1, int id2, int sz2) {
  std::vector<int> ans(sz1 + sz2);
  int save_id1 = id1, save_id2 = id2;
  int k1 = sz1;
  int k2 = sz2;
  for (int i = 0; i < sz1 + sz2; ++i) {
    if (k1 != 0 && k2 != 0) {
      if (mass_[id1] < mass_[id2]) {
        --k1;
        ans[i] = mass_[id1];
        ++id1;
      } else {
        --k2;
        ans[i] = mass_[id2];
        ++id2;
      }
    } else {
      if (k1) {
        ans[i] = mass_[id1];
        ++id1;
      } else {
        ans[i] = mass_[id2];
        ++id2;
      }
    }
  }

  for (int i = 0; i < sz1; ++i) {
    mass_[i + save_id1] = ans[i];
  }

  for (int i = 0; i < sz2; ++i) {
    mass_[i + save_id2] = ans[i + sz1];
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::FindThreadVariables() {
  c_threads_ = omp_get_num_threads();
  mini_batch_ = 2 * c_threads_;

  if (size_ % mini_batch_ != 0) {
    mass_.resize(size_ + (size_ % mini_batch_));
    int max_elem = std::numeric_limits<int>::min();
    for (int i = 0; i < size_; ++i) {
      mass_[i] = input_[i];
      max_elem = std::max(max_elem, input_[i]);
    }

    for (int i = size_; i < std:: : static_cast<int>(mass_.size()); ++i) {
      mass_[i] = max_elem;
    }
  } else {
    mass_ = input_;
  }
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::RunImpl() {
  ParallelShellSort(array_, array_.size());
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PostProcessingImpl() {
  for (size_t i = 0; i < array_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = array_[i];
  }
  return true;
}