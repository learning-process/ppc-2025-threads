// Copyright Anikin Maksim 2025
#include "seq/anikin_m_shall_batcher/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

void anikin_m_shall_batcher_seq::ShellSort(std::vector<int>& arr) {
  int n = static_cast<int>(arr.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = arr[i];
      int j = 0;
      for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
        arr[j] = arr[j - gap];
      }
      arr[j] = temp;
    }
  }
}

void anikin_m_shall_batcher_seq::BatcherOddEvenMerge(std::vector<int>& arr1, std::vector<int>& arr2,
                                                     std::vector<int>& output) {
  int i = 0;
  int j = 0;
  int k = 0;
  int n1 = static_cast<int>(arr1.size());
  int n2 = static_cast<int>(arr2.size());

  while (i < n1 && j < n2) {
    if (arr1[i] < arr2[j]) {
      output[k++] = arr1[i++];
    } else {
      output[k++] = arr2[j++];
    }
  }

  while (i < n1) {
    output[k++] = arr1[i++];
  }

  while (j < n2) {
    output[k++] = arr2[j++];
  }
}

void anikin_m_shall_batcher_seq::ShellSortWithBatcherMerge(const std::vector<int>& input, std::vector<int>& output) {
  if (input.empty()) {
    output.clear();
    return;
  }

  std::vector<int> arr1(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(input.size() / 2));
  std::vector<int> arr2(input.begin() + static_cast<std::ptrdiff_t>(input.size() / 2), input.end());

  ShellSort(arr1);
  ShellSort(arr2);

  output.resize(input.size());
  BatcherOddEvenMerge(arr1, arr2, output);
}

bool anikin_m_shall_batcher_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool anikin_m_shall_batcher_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool anikin_m_shall_batcher_seq::TestTaskSequential::RunImpl() {
  ShellSortWithBatcherMerge(input_, output_);
  return true;
}

bool anikin_m_shall_batcher_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}