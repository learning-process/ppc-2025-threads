#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

#include <tbb/parallel_invoke.h>

#include <algorithm>

namespace shuravina_o_hoare_simple_merger_tbb {

HoareSortTBB::HoareSortTBB(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

bool HoareSortTBB::Validation() {
  if (!task_data) return false;
  return task_data->inputs_count.size() == 1 && task_data->outputs_count.size() == 1 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool HoareSortTBB::PreProcessing() {
  try {
    if (!task_data || task_data->inputs.empty() || task_data->inputs_count.empty()) {
      return false;
    }

    auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
    data_ = std::vector<int>(input_data, input_data + task_data->inputs_count[0]);
    return true;
  } catch (...) {
    return false;
  }
}

bool HoareSortTBB::Run() {
  if (data_.empty()) return false;

  try {
    ParallelQuickSort(data_.data(), 0, data_.size() - 1);
    return true;
  } catch (...) {
    return false;
  }
}

bool HoareSortTBB::PostProcessing() {
  try {
    if (!task_data || task_data->outputs.empty()) {
      return false;
    }

    auto* output_data = reinterpret_cast<int*>(task_data->outputs[0]);
    std::copy(data_.begin(), data_.end(), output_data);
    return true;
  } catch (...) {
    return false;
  }
}

size_t HoareSortTBB::Partition(int* arr, size_t left, size_t right) {
  int pivot = arr[(left + right) / 2];
  while (left <= right) {
    while (arr[left] < pivot) ++left;
    while (arr[right] > pivot) --right;
    if (left <= right) {
      std::swap(arr[left], arr[right]);
      ++left;
      --right;
    }
  }
  return left;
}

void HoareSortTBB::SequentialQuickSort(int* arr, size_t left, size_t right) {
  if (left >= right) return;

  size_t p = Partition(arr, left, right);
  if (left < p - 1) SequentialQuickSort(arr, left, p - 1);
  if (p < right) SequentialQuickSort(arr, p, right);
}

void HoareSortTBB::ParallelQuickSort(int* arr, size_t left, size_t right) {
  if (right - left < kThreshold) {
    SequentialQuickSort(arr, left, right);
    return;
  }

  size_t p = Partition(arr, left, right);

  if (left < p - 1 && p < right) {
    tbb::parallel_invoke([&] { ParallelQuickSort(arr, left, p - 1); }, [&] { ParallelQuickSort(arr, p, right); });
  } else if (left < p - 1) {
    ParallelQuickSort(arr, left, p - 1);
  } else if (p < right) {
    ParallelQuickSort(arr, p, right);
  }
}

}  // namespace shuravina_o_hoare_simple_merger_tbb