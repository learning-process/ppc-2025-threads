#include "omp/shuravina_o_hoare_simple_merger_omp1/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace shuravina_o_hoare_simple_merger {

void TestTaskOMP::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int mid = low + (high - low) / 2;
    if (arr[high] < arr[low]) std::swap(arr[low], arr[high]);
    if (arr[mid] < arr[low]) std::swap(arr[mid], arr[low]);
    if (arr[high] < arr[mid]) std::swap(arr[mid], arr[high]);
    int pivot = arr[mid];

    int i = low;
    int j = high;

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
      if (low < j) QuickSort(arr, low, j);
#pragma omp section
      if (i < high) QuickSort(arr, i, high);
    }
  }
}

void TestTaskOMP::Merge(std::vector<int>& arr, int low, int mid, int high) {
  static thread_local std::vector<int> temp;
  temp.resize(high - low + 1);

  int i = low;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= high) {
    temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
  }

  while (i <= mid) temp[k++] = arr[i++];
  while (j <= high) temp[k++] = arr[j++];

  const size_t block_size = 64 / sizeof(int);
#pragma omp parallel for
  for (size_t idx = 0; idx <= high - low; idx += block_size) {
    size_t end = std::min(idx + block_size, static_cast<size_t>(high - low + 1));
    std::copy(temp.begin() + idx, temp.begin() + end, arr.begin() + low + idx);
  }
}

bool TestTaskOMP::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
    return false;
  }

  const auto* in_ptr = reinterpret_cast<const int*>(task_data->inputs[0]);
  const size_t input_size = task_data->inputs_count[0];

  input_.reserve(input_size);
  input_.assign(in_ptr, in_ptr + input_size);

  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  const size_t output_size = task_data->outputs_count[0];
  output_.resize(output_size);

  return true;
}

bool TestTaskOMP::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskOMP::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  omp_set_dynamic(0);

  const int size = static_cast<int>(input_.size());
  QuickSort(input_, 0, size - 1);

  if (omp_get_max_threads() > 1) {
    Merge(input_, 0, size / 2 - 1, size - 1);
  }

  output_ = std::move(input_);
  return true;
}

bool TestTaskOMP::PostProcessingImpl() {
  if (output_.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  const size_t block_size = 64 / sizeof(int);
#pragma omp parallel for
  for (size_t i = 0; i < output_.size(); i += block_size) {
    size_t end = std::min(i + block_size, output_.size());
    std::copy(output_.begin() + i, output_.begin() + end, out_ptr + i);
  }

  return true;
}

}  // namespace shuravina_o_hoare_simple_merger