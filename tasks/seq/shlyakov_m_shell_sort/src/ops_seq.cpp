#include "seq/example/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool shlyakov_m_shell_sort_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

bool TestTaskSequential::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool shlyakov_m_shell_sort_seq::TestTaskSequential::RunImpl() {
  int n = input_.size();

  std::vector<int> gaps;
  for (int i = 1; i <= static_cast<int>(sqrt(n)) + 1; ++i) {
    int gap = n / (int)pow(2, i);
    if (gap > 0) {
      gaps.push_back(gap);
    }
  }

  for (int k = gaps.size() - 1; k >= 0; --k) {
    int gap = gaps[k];
    for (int start = 0; start < gap; ++start) {
      for (int i = start + gap; i < n; i += gap) {
        int key = output_[i];
        int j = i - gap;
        while (j >= start && output_[j] > key) {
          output_[j + gap] = output_[j];
          j -= gap;
        }
        output_[j + gap] = key;
      }
    }

    //  for (int i = 0; i < n - gap; ++i) {
    //    if (output_[i] > output_[i + gap]) {
    //      swap(output_[i], output_[i + gap]);
    //    }
    //  }
  }
}

bool shlyakov_m_shell_sort_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
