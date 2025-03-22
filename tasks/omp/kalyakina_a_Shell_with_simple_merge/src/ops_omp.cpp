#include "omp/kalyakina_a_Shell_with_simple_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

std::vector<unsigned int> kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::CalculationOfGapLengths(
    const unsigned int &size) {
  std::vector<unsigned int> result;
  unsigned int local_res = 1;
  for (unsigned int i = 1; (local_res * 3 <= size) || (local_res == 1); i++) {
    result.push_back(local_res);
    if (i % 2 != 0) {
      local_res = (unsigned int)((8 * pow(2, i)) - (6 * pow(2, (float)(i + 1) / 2)) + 1);
    } else {
      local_res = (unsigned int)((9 * pow(2, i)) - (9 * pow(2, (float)i / 2)) + 1);
    }
  }
  return result;
}

void kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::ShellSort(std::vector<int> &vec, unsigned int &left,
                                                                         unsigned int &right) {
  for (unsigned int k = Sedgwick_sequence_.size(); k > 0;) {
    unsigned int gap = Sedgwick_sequence_[--k];
    for (unsigned int i = left; i < left + gap; i++) {
      for (unsigned int j = i; j < right; j += gap) {
        unsigned int index = j;
        int tmp = vec[index];
        while ((index >= i + gap) && (tmp < vec[index - gap])) {
          vec[index] = vec[index - gap];
          index -= gap;
        }
        vec[index] = tmp;
      }
    }
  }
}

void kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::SimpleMergeSort(std::vector<int> &vec,
                                                                               unsigned int &left, unsigned int &middle,
                                                                               unsigned int &right) {
  std::vector<int> first_part(middle - left);
  std::copy(vec.begin() + left, vec.begin() + middle, first_part.begin());
  unsigned int l = 0;
  unsigned int r = middle;
  unsigned int j = left;
  for (; (l < first_part.size()) && (r < right); j++) {
    if (first_part[l] < vec[r]) {
      vec[j] = first_part[l++];
    } else {
      vec[j] = vec[r++];
    }
  }
  while (l < first_part.size()) {
    vec[j++] = first_part[l++];
  }
  while (r < right) {
    vec[j++] = vec[r++];
  }
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  std::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());

  return true;
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::ValidationImpl() {
  return (task_data->inputs_count[0] > 0) && (task_data->outputs_count[0] > 0) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::RunImpl() {
  std::vector<std::pair<unsigned int, unsigned int>> bounds;
  unsigned int num =
      ((unsigned int)omp_get_max_threads() > input_.size()) ? input_.size() : (unsigned int)omp_get_max_threads();
  unsigned int part = input_.size() / num;
  unsigned int reminder = input_.size() % num;
  unsigned int left = 0;
  Sedgwick_sequence_ = CalculationOfGapLengths(input_.size() / num);
  for (unsigned int i = 0; i < num; i++) {
    unsigned int right = (i < reminder) ? left + part + 1 : left + part;
    bounds.emplace_back(left, right);
    left = right;
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < (int)num; i++) {
    ShellSort(input_, bounds[i].first, bounds[i].second);
  }
  num = std::ceil((double)num / 2);
  unsigned int step = 1;
  while (step < bounds.size()) {
    step *= 2;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)num; i++) {
      unsigned int middle = (step / 2) + (step * i);
      if (middle < bounds.size()) {
        SimpleMergeSort(
            input_, bounds[i * step].first, bounds[middle].first,
            bounds[(bounds.size() - 1 < (i + 1) * step - 1) ? bounds.size() - 1 : ((i + 1) * step) - 1].second);
      }
    }
    num = std::ceil((double)num / 2);
  }
  return true;
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
