#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "oneapi/tbb/parallel_invoke.h"

namespace shuravina_o_hoare_simple_merger_tbb {

size_t TestTaskTBB::Partition(std::vector<double>& arr, size_t low, size_t high) {
  double pivot = arr[high];
  size_t i = low;

  for (size_t j = low; j < high; ++j) {
    if (arr[j] <= pivot) {
      std::swap(arr[i], arr[j]);
      ++i;
    }
  }
  std::swap(arr[i], arr[high]);
  return i;
}

void TestTaskTBB::QuickSort(std::vector<double>& arr, size_t low, size_t high) {
  if (low < high) {
    size_t pi = Partition(arr, low, high);
    if (pi > 0) QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

void TestTaskTBB::ParallelQuickSort(std::vector<double>& arr, size_t low, size_t high) {
  if (low < high) {
    if (high - low < kParallelThreshold) {
      QuickSort(arr, low, high);
      return;
    }

    size_t pi = Partition(arr, low, high);

    tbb::parallel_invoke([&] { ParallelQuickSort(arr, low, pi > 0 ? pi - 1 : 0); },
                         [&] { ParallelQuickSort(arr, pi + 1, high); });
  }
}

bool TestTaskTBB::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto input_size = task_data->inputs_count[0];
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  output_ = std::vector<double>(input_size, 0);
  chunk_count_ = task_data->inputs_count[1];

  return true;
}

bool TestTaskTBB::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->outputs.empty() && task_data->inputs_count.size() >= 2 &&
         task_data->inputs_count[0] > 2 && task_data->inputs_count[1] >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskTBB::RunImpl() {
  auto size = input_.size();
  if (size > 0) {
    if (size < kParallelThreshold) {
      QuickSort(input_, 0, size - 1);
    } else {
      ParallelQuickSort(input_, 0, size - 1);
    }
    output_ = input_;
  }
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  if (output_.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_tbb