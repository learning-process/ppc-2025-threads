#include "seq/dudchenko_o_hoare_batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool dudchenko_o_hoare_batcher_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool dudchenko_o_hoare_batcher_seq::TestTaskSequential::ValidationImpl() {
  if (!task_data || (task_data->inputs[0] == nullptr) || (task_data->outputs[0] == nullptr)) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool dudchenko_o_hoare_batcher_seq::TestTaskSequential::RunImpl() {
  if (input_.empty()) {
    return false;
  }

  QuickSort(input_, 0, static_cast<int>(input_.size()) - 1);
  output_ = input_;
  return true;
}

bool dudchenko_o_hoare_batcher_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void dudchenko_o_hoare_batcher_seq::TestTaskSequential::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pi = Partition(arr, low, high);
    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

int dudchenko_o_hoare_batcher_seq::TestTaskSequential::Partition(std::vector<int>& arr, int low, int high) {
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j <= high - 1; j++) {
    if (arr[j] < pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return (i + 1);
}

void dudchenko_o_hoare_batcher_seq::TestTaskSequential::BatcherMerge(std::vector<int>& arr, int left, int mid,
                                                                     int right) {
  int n1 = mid - left + 1;
  int n2 = right - mid;

  std::vector<int> left_array(n1);
  std::vector<int> right_array(n2);

  for (int i = 0; i < n1; i++) {
    left_array[i] = arr[left + i];
  }
  for (int j = 0; j < n2; j++) {
    right_array[j] = arr[mid + 1 + j];
  }

  int i = 0;
  int j = 0;
  int k = left;

  while (i < n1 && j < n2) {
    if (left_array[i] <= right_array[j]) {
      arr[k] = left_array[i];
      i++;
    } else {
      arr[k] = right_array[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    arr[k] = left_array[i];
    i++;
    k++;
  }

  while (j < n2) {
    arr[k] = right_array[j];
    j++;
    k++;
  }
}