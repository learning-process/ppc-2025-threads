#include "omp/shuravina_o_hoare_simple_merger_omp1/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace shuravina_o_hoare_simple_merger {

void TestTaskOMP::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    if (high - low < 100) {
      std::sort(arr.begin() + low, arr.begin() + high + 1);
      return;
    }

    int pivot = arr[high];
    int i = low - 1;

#pragma omp parallel for shared(arr)
    for (int j = low; j < high; ++j) {
      if (arr[j] <= pivot) {
#pragma omp atomic
        i++;
        std::swap(arr[i], arr[j]);
      }
    }
    std::swap(arr[i + 1], arr[high]);

    int pi = i + 1;

#pragma omp task shared(arr) if (high - low > 10000)
    QuickSort(arr, low, pi - 1);

#pragma omp task shared(arr) if (high - low > 10000)
    QuickSort(arr, pi + 1, high);
  }
}

void TestTaskOMP::Merge(std::vector<int>& arr, int low, int mid, int high) {
  static thread_local std::vector<int> temp;
  temp.resize(high - low + 1);

  int i = low;
  int j = mid + 1;
  int k = 0;

#pragma omp parallel sections
  {
#pragma omp section
    {
      while (i <= mid && j <= high) {
        temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
      }
    }

#pragma omp section
    {
      while (i <= mid) {
        temp[k++] = arr[i++];
      }
    }

#pragma omp section
    {
      while (j <= high) {
        temp[k++] = arr[j++];
      }
    }
  }

#pragma omp parallel for
  for (int idx = 0; idx <= high - low; ++idx) {
    arr[low + idx] = temp[idx];
  }
}

bool TestTaskOMP::PreProcessingImpl() {
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  const int* in_ptr = reinterpret_cast<const int*>(task_data->inputs[0]);
  const size_t input_size = task_data->inputs_count[0];

  input_.assign(in_ptr, in_ptr + input_size);
  output_.resize(task_data->outputs_count[0]);

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
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  return true;
}

bool TestTaskOMP::RunImpl() {
  if (input_.empty()) {
    return true;
  }

#pragma omp parallel
  {
#pragma omp single
    QuickSort(input_, 0, static_cast<int>(input_.size()) - 1);
  }

  output_ = std::move(input_);

  return true;
}

bool TestTaskOMP::PostProcessingImpl() {
  if (output_.empty()) {
    return true;
  }

  int* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  const size_t count = output_.size();
  const size_t bytes = count * sizeof(int);

  const bool no_overlap =
      (reinterpret_cast<const uint8_t*>(output_.data()) + bytes <= reinterpret_cast<const uint8_t*>(out_ptr)) ||
      (reinterpret_cast<uint8_t*>(out_ptr) + bytes <= reinterpret_cast<const uint8_t*>(output_.data()));

  if (no_overlap) {
    std::memcpy(out_ptr, output_.data(), bytes);
  } else {
    std::ranges::copy(output_, out_ptr);
  }

  return true;
}

}  // namespace shuravina_o_hoare_simple_merger