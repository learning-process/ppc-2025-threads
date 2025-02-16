#include "seq/korovin_n_qsort_batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

int korovin_n_qsort_batcher_seq::TestTaskSequential::GetRandomIndex(int low, int high) {
  return low + (std::rand() % (high - low + 1));
}

void korovin_n_qsort_batcher_seq::TestTaskSequential::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low >= high) {
    return;
  }

  int partition_index = GetRandomIndex(low, high);
  int partition_value = arr[partition_index];
  int i = low;
  int j = high;

  while (i <= j) {
    while (arr[i] < partition_value) {
      i++;
    }
    while (arr[j] > partition_value) {
      j--;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

  if (j > low) {
    QuickSort(arr, low, j);
  }
  if (i < high) {
    QuickSort(arr, i, high);
  }
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  return true;
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::ValidationImpl() {
  return (!task_data->inputs.empty()) && (!task_data->outputs.empty()) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n <= 1) {
    return true;
  }
  QuickSort(input_, 0, n - 1);
  return true;
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(input_.begin(), input_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
