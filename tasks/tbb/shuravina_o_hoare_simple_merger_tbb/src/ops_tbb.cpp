#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "oneapi/tbb/parallel_invoke.h"

namespace shuravina_o_hoare_simple_merger_tbb {

void TestTaskTBB::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < 0 || high < 0 || low >= arr.size() || high >= arr.size() || arr.empty()) {
    return;
  }

  if (low < high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
      if (arr[j] <= pivot) {
        ++i;
        std::swap(arr[i], arr[j]);
      }
    }
    std::swap(arr[i + 1], arr[high]);

    int pi = i + 1;

    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

void TestTaskTBB::ParallelQuickSort(std::vector<int>& arr, int low, int high) {
  if (low < 0 || high < 0 || low >= arr.size() || high >= arr.size() || arr.empty()) {
    return;
  }

  if (high - low < 1000) {
    QuickSort(arr, low, high);
    return;
  }

  int pivot = arr[high];
  int i = low - 1;

  for (int j = low; j < high; ++j) {
    if (arr[j] <= pivot) {
      ++i;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);

  int pi = i + 1;

  tbb::parallel_invoke([&] { ParallelQuickSort(arr, low, pi - 1); }, [&] { ParallelQuickSort(arr, pi + 1, high); });
}

void TestTaskTBB::Merge(std::vector<int>& arr, int low, int mid, int high) {
  std::vector<int> temp(high - low + 1);
  int i = low;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= high) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }

  while (j <= high) {
    temp[k++] = arr[j++];
  }

  for (i = low, k = 0; i <= high; ++i, ++k) {
    arr[i] = temp[k];
  }
}

bool TestTaskTBB::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  auto input_size = task_data->inputs_count[0];
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  auto output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool TestTaskTBB::ValidationImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskTBB::RunImpl() {
  try {
    auto size = input_.size();
    if (size < 10000) {
      QuickSort(input_, 0, static_cast<int>(size) - 1);
    } else {
      ParallelQuickSort(input_, 0, static_cast<int>(size) - 1);
    }
    Merge(input_, 0, static_cast<int>(size / 2) - 1, static_cast<int>(size) - 1);
    output_ = input_;
    return true;
  } catch (...) {
    return false;
  }
}

bool TestTaskTBB::PostProcessingImpl() {
  if (!task_data || output_.empty() || task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_tbb