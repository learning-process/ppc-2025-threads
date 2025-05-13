#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <vector>

#include "oneapi/tbb/parallel_invoke.h"

namespace shuravina_o_hoare_simple_merger_tbb {

int TestTaskTBB::Partition(std::vector<int>& arr, int low, int high) {
  int pivot = arr[high];
  int i = low - 1;

  for (int j = low; j < high; ++j) {
    if (arr[j] <= pivot) {
      ++i;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return i + 1;
}

void TestTaskTBB::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pi = Partition(arr, low, high);
    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

void TestTaskTBB::ParallelQuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    if (high - low < static_cast<int>(kParallelThreshold)) {
      QuickSort(arr, low, high);
      return;
    }

    int pi = Partition(arr, low, high);

    tbb::parallel_invoke([&] { ParallelQuickSort(arr, low, pi - 1); }, [&] { ParallelQuickSort(arr, pi + 1, high); });
  }
}

bool TestTaskTBB::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  auto input_size = task_data->inputs_count[0];
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  output_ = std::vector<int>(input_size, 0);

  return true;
}

bool TestTaskTBB::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskTBB::RunImpl() {
  auto size = input_.size();
  if (size > 0) {
    if (size < kParallelThreshold) {
      QuickSort(input_, 0, static_cast<int>(size) - 1);
    } else {
      ParallelQuickSort(input_, 0, static_cast<int>(size) - 1);
    }
    output_ = input_;
  }
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  if (output_.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_tbb