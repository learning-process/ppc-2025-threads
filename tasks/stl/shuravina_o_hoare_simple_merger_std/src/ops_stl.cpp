#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

#include <omp.h>

#include <algorithm>
#include <thread>
#include <vector>

namespace shuravina_o_hoare_simple_merger_stl {

bool TestTaskSTL::PreProcessingImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void TestTaskSTL::QuickSort(std::vector<int>& arr, int left, int right) {
  if (left >= right) return;

  int pivot = arr[(left + right) / 2];
  int i = left, j = right;

  while (i <= j) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

  QuickSort(arr, left, j);
  QuickSort(arr, i, right);
}

void TestTaskSTL::ParallelQuickSort(std::vector<int>& arr, int left, int right) {
  if (left >= right) return;

  int pivot = arr[(left + right) / 2];
  int i = left, j = right;

  while (i <= j) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    { ParallelQuickSort(arr, left, j); }

#pragma omp section
    { ParallelQuickSort(arr, i, right); }
  }
}

void TestTaskSTL::Merge(std::vector<int>& arr, int left, int mid, int right) {
  std::vector<int> temp(right - left + 1);
  int i = left, j = mid + 1, k = 0;

  while (i <= mid && j <= right) {
    temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) temp[k++] = arr[i++];
  while (j <= right) temp[k++] = arr[j++];

  std::copy(temp.begin(), temp.end(), arr.begin() + left);
}

bool TestTaskSTL::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  ParallelQuickSort(input_, 0, input_.size() - 1);

  Merge(input_, 0, input_.size() / 2 - 1, input_.size() - 1);

  output_ = input_;
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  if (output_.empty()) return true;

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_stl