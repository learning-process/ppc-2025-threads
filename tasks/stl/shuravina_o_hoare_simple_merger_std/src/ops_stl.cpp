#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
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
  return task_data->inputs_count.size() >= 2 && task_data->inputs_count[0] > 2 && task_data->inputs_count[1] >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskSTL::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  input_ = *reinterpret_cast<std::vector<double>*>(task_data->inputs[0]);
  output_.resize(task_data->outputs_count[0]);
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = input_.size() / chunk_count_;
  return true;
}

void TestTaskSTL::QuickSort(std::vector<double>& arr, size_t left, size_t right) {
  if (left >= right) {
    return;
  }

  const double pivot = arr[(left + right) / 2];
  size_t i = left;
  size_t j = right;

  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      if (j > 0) {
        j--;
      } else {
        break;
      }
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      if (j > 0) {
        j--;
      }
    }
  }

  if (j > left) {
    QuickSort(arr, left, j);
  }
  if (i < right) {
    QuickSort(arr, i, right);
  }
}

void TestTaskSTL::MergeHelper(std::vector<double>& arr, size_t left, size_t mid, size_t right) {
  std::vector<double> temp(right - left + 1);
  size_t i = left;
  size_t j = mid + 1;
  size_t k = 0;

  while (i <= mid && j <= right) {
    temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= right) {
    temp[k++] = arr[j++];
  }

  std::ranges::copy(temp, arr.begin() + static_cast<long>(left));
}

bool TestTaskSTL::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  const size_t num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> workers;

  if (chunk_count_ < num_threads) {
    chunk_count_ = num_threads;
    min_chunk_size_ = input_.size() / chunk_count_;
  }

  for (size_t i = 0; i < chunk_count_; ++i) {
    const size_t start = i * min_chunk_size_;
    const size_t end = (i == chunk_count_ - 1) ? (input_.size() - 1) : ((i + 1) * min_chunk_size_ - 1);
    workers.emplace_back([this, start, end]() { QuickSort(input_, start, end); });
  }
  for (auto& t : workers) {
    t.join();
  }
  workers.clear();

  for (size_t merge_size = min_chunk_size_; merge_size < input_.size(); merge_size *= 2) {
    for (size_t left = 0; left < input_.size(); left += (2 * merge_size)) {
      const size_t mid = left + merge_size - 1;
      if (mid >= input_.size() - 1) {
        break;
      }
      const size_t right = std::min(left + (2 * merge_size) - 1, input_.size() - 1);
      workers.emplace_back([this, left, mid, right]() { MergeHelper(input_, left, mid, right); });
    }
    for (auto& t : workers) {
      t.join();
    }
    workers.clear();
  }

  output_ = input_;
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  if (output_.empty()) {
    return true;
  }
  *reinterpret_cast<std::vector<double>*>(task_data->outputs[0]) = output_;
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_stl