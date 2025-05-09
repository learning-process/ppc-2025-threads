#include "stl/deryabin_m_hoare_sort_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <stack>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void deryabin_m_hoare_sort_simple_merge_stl::HoaraSort(std::vector<double>& a, size_t first, size_t last) {
  std::stack<std::pair<size_t, size_t>> stack;
  stack.push({first, last});
  while (!stack.empty()) {
    auto [first, last] = stack.top();
    stack.pop();
    if (first >= last) {
      continue;
    }
    size_t i = first;
    size_t j = last;
    double tmp = 0;
    double x = std::max(std::min(a[first], a[(first + last) / 2]),
                        std::min(std::max(a[first], a[(first + last) / 2]), a[last]));
    do {
      while (a[i] < x) {
        i++;
      }
      while (a[j] > x) {
        j--;
      }
      if (i < j && a[i] > a[j]) {
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
      }
    } while (i < j);
    stack.push({i + 1, last});
    stack.push({first, j});
  }
}

void MergeTwoParts(std::vector<double>& arr, size_t left, size_t right) {
  size_t mid = left + (right - left) / 2;
  std::inplace_merge(arr.begin() + left, arr.begin() + mid + 1, arr.begin() + right + 1);
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  size_t chunk_count = chunk_count_;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  for (size_t i = 0; i < (size_t)(log((double)chunk_count_) / std::numbers::ln2); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      MergeTwoParts(input_array_A_, j * min_chunk_size_ << (i + 1), ((j + 1) * min_chunk_size_ << (i + 1)) - 1,
                    dimension_);
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::RunImpl() {
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> workers;
  workers.reserve(num_threads); 
  auto parallel_for = [&workers, num_threads](size_t start, size_t end, const std::function<void(size_t)>& func) {
    const size_t total = end - start;
    const size_t chunk_size = std::max<size_t>(1, total / num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      const size_t chunk_start = start + i * chunk_size;
      const size_t chunk_end = (i == num_threads - 1) ? end : chunk_start + chunk_size;
      workers.emplace_back([=, &func] {
        for (size_t j = chunk_start; j < chunk_end; ++j) {
          func(j);
        }
      });
    }
    for (auto& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers.clear();
  };
  parallel_for(0, chunk_count_, [this](size_t count) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
  });
  const int merge_steps = static_cast<int>(std::log2(chunk_count_));
  for (int i = 0; i < merge_steps; ++i) {
    const size_t chunks_per_step = chunk_count_ >> (i + 1);
    parallel_for(0, chunks_per_step, [this, i](size_t j) {
      const size_t left = j * (min_chunk_size_ << (i + 1));
      const size_t right = ((j + 1) * (min_chunk_size_ << (i + 1))) - 1;
      MergeTwoParts(input_array_A_, left, right, dimension_);
    });
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
