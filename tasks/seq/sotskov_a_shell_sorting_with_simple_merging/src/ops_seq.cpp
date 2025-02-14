#include "seq/sotskov_a_shell_sorting_with_simple_merging/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <thread>
#include <vector>

std::vector<int> sotskov_a_shell_sorting_with_simple_merging_seq::shellSort(const std::vector<int>& inputArray) {
  std::vector<int> sortedArray = inputArray;
  int arraySize = sortedArray.size();

  std::vector<int> gapSequence;
  int currentGap = 1;
  while (currentGap < arraySize / 3) {
    gapSequence.push_back(currentGap);
    currentGap = currentGap * 3 + 1;
  }

  for (int gapIndex = gapSequence.size() - 1; gapIndex >= 0; --gapIndex) {
    int gap = gapSequence[gapIndex];
    for (int i = gap; i < arraySize; ++i) {
      int currentElement = sortedArray[i];
      int j = i;
      while (j >= gap && sortedArray[j - gap] > currentElement) {
        sortedArray[j] = sortedArray[j - gap];
        j -= gap;
      }
      sortedArray[j] = currentElement;
    }
  }

  return sortedArray;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = tempPtr[i];
  }

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  int inputSize = task_data->inputs_count[0];
  int outputSize = task_data->outputs_count[0];

  return (inputSize == outputSize);
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::RunImpl() {
  result_ = shellSort(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  int* output_ = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(result_.begin(), result_.end(), output_);
  return true;
}
