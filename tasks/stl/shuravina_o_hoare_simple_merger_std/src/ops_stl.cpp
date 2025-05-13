#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_stl {

TestTaskSTL::TestTaskSTL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

bool TestTaskSTL::Validation() { return ValidationImpl(); }
bool TestTaskSTL::PreProcessing() { return PreProcessingImpl(); }
bool TestTaskSTL::Run() { return RunImpl(); }
bool TestTaskSTL::PostProcessing() { return PostProcessingImpl(); }

bool TestTaskSTL::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->outputs.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskSTL::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  int* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  return true;
}

void TestTaskSTL::QuickSort(std::vector<int>& arr, int left, int right) {
  if (left >= right) {
    return;
  }

  int pivot = arr[(left + right) / 2];
  int i = left;
  int j = right;

  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

  QuickSort(arr, left, j);
  QuickSort(arr, i, right);
}

void TestTaskSTL::MergeHelper(std::vector<int>& arr, int left, int mid, int right) {
  std::vector<int> temp(right - left + 1);
  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= right) {
    temp[k++] = arr[j++];
  }

  std::ranges::copy(temp, arr.begin() + left);
}
bool TestTaskSTL::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  const int size = static_cast<int>(input_.size());
  QuickSort(input_, 0, size - 1);
  MergeHelper(input_, 0, (size / 2) - 1, size - 1);
  output_ = input_;
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  if (output_.empty()) {
    return true;
  }
  int* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_stl