#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

#include <algorithm>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

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

  int mid = left + ((right - left) / 2);
  if (arr[mid] < arr[left]) {
    std::swap(arr[left], arr[mid]);
  }
  if (arr[right] < arr[left]) {
    std::swap(arr[left], arr[right]);
  }
  if (arr[mid] < arr[right]) {
    std::swap(arr[mid], arr[right]);
  }
  
  int pivot = arr[right];
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
  const int num_threads = ppc::util::GetPPCNumThreads();
  const int chunk_size = size / num_threads;

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == (num_threads - 1)) ? (size - 1) : (((i + 1) * chunk_size) - 1);
    threads.emplace_back([this, start, end]() { QuickSort(input_, start, end); });
  }

  for (auto& t : threads) {
    t.join();
  }
  threads.clear();

  for (int merge_size = chunk_size; merge_size < size; merge_size *= 2) {
    for (int left = 0; left < size; left += (2 * merge_size)) {
      int mid = left + merge_size - 1;
      if (mid >= (size - 1)) {
        break;
      }
      int right = std::min(left + (2 * merge_size) - 1, size - 1);
      MergeHelper(input_, left, mid, right);
    }
  }

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