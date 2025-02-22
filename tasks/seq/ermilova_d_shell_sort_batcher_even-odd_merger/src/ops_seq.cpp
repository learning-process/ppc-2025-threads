#include "seq/ermilova_d_shell_sort_batcher_even-odd_merger/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

std::vector<int> ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::SedgwickSequence(int n) {
  std::vector<int> gaps;
  int k = 0;
  while (true) {
    int gap;
    if (k % 2 == 0) {
      gap = 9 * (1 << (2 * k)) - 9 * (1 << k) + 1;
    } else {
      gap = 8 * (1 << k) - 6 * (1 << ((k + 1) / 2)) + 1;
    }

    if (gap >= n) break;

    gaps.push_back(gap);
    k++;
  }
  return gaps;
}

void ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::ShellSort(std::vector<int> &vec) {
  int n = vec.size();
  std::vector<int> gaps = SedgwickSequence(n);

  for (int k = gaps.size() - 1; k >= 0; k--) {
    int gap = gaps[k];
    for (int i = gap; i < n; i++) {
      int temp = vec[i];
      int j;
      for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return (task_data->inputs_count[0] > 0) && (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::RunImpl() {
  ShellSort(input_);
  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::PostProcessingImpl() {
  auto *data = reinterpret_cast<int *>(task_data->outputs[0]);
  std::copy(input_.begin(), input_.end(), data);
  return true;
}
