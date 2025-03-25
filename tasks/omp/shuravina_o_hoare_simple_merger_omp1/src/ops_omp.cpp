#include "omp/shuravina_o_hoare_simple_merger_omp1/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
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
  thread_local std::vector<int> temp;
  temp.resize(high - low + 1);

  int i = low;
  int j = mid + 1;
  int k = 0;

#pragma omp parallel sections
  {
#pragma omp section
    {
      while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) {
          temp[k++] = arr[i++];
        } else {
          temp[k++] = arr[j++];
        }
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
  if (!task_data->inputs[0] || !task_data->outputs[0]) {
    return false;
  }

  input_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);

  output_.clear();
  output_.reserve(task_data->outputs_count[0]);

  return true;
}

bool TestTaskOMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs[0] != nullptr &&
         task_data->outputs[0] != nullptr;
}

bool TestTaskOMP::RunImpl() {
  if (input_.empty()) return true;

#pragma omp parallel
  {
#pragma omp single
    QuickSort(input_, 0, static_cast<int>(input_.size()) - 1);
  }

  output_ = std::move(input_);

  return true;
}

bool TestTaskOMP::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], output_.data(), output_.size() * sizeof(int));
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger