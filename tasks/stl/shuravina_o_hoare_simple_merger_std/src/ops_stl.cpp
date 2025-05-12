#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

#include <algorithm>
#include <execution>
#include <future>
#include <memory>
#include <thread>
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

  int mid = left + (right - left) / 2;
  if (arr[left] > arr[mid]) std::swap(arr[left], arr[mid]);
  if (arr[left] > arr[right]) std::swap(arr[left], arr[right]);
  if (arr[mid] > arr[right]) std::swap(arr[mid], arr[right]);
  int pivot = arr[mid];
  std::swap(arr[mid], arr[right - 1]);

  int i = left;
  int j = right - 1;
  while (true) {
    while (arr[++i] < pivot) {
    }
    while (arr[--j] > pivot) {
    }
    if (i >= j) break;
    std::swap(arr[i], arr[j]);
  }
  std::swap(arr[i], arr[right - 1]);

  const int parallel_threshold = 5000;
  if (right - left > parallel_threshold) {
    auto future = std::async(std::launch::async, [&arr, left, i]() { QuickSort(arr, left, i - 1); });
    QuickSort(arr, i + 1, right);
    future.get();
  } else {
    QuickSort(arr, left, i - 1);
    QuickSort(arr, i + 1, right);
  }
}

void TestTaskSTL::MergeHelper(std::vector<int>& arr, int left, int mid, int right) {
  std::vector<int> temp(right - left + 1);

  const int parallel_threshold = 10000;
  if (right - left > parallel_threshold) {
    auto future = std::async(std::launch::async, [&arr, &temp, left, mid, right]() {
      int i = left;
      int j = mid + 1;
      int k = 0;
      while (i <= mid && j <= mid + (right - left) / 2) {
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
      }
      while (i <= mid) temp[k++] = arr[i++];
    });

    int i = mid + 1 + (right - left) / 2;
    int j = mid + 1;
    int k = (right - left) / 2 + 1;
    while (i <= right && j <= right) {
      temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= right) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    future.get();
  } else {
    int i = left;
    int j = mid + 1;
    int k = 0;
    while (i <= mid && j <= right) {
      temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
  }

  const int copy_threshold = 5000;
  if (temp.size() > copy_threshold) {
    std::copy(std::execution::par, temp.begin(), temp.end(), arr.begin() + left);
  } else {
    std::copy(temp.begin(), temp.end(), arr.begin() + left);
  }
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

  if (output_.size() > 5000) {
    std::copy(std::execution::par, output_.begin(), output_.end(), out_ptr);
  } else {
    std::copy(output_.begin(), output_.end(), out_ptr);
  }

  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_stl