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

void deryabin_m_hoare_sort_simple_merge_tbb::HoaraSort(std::vector<double>& a, size_t first, size_t last) {
  if (first >= last) {
    return;
  }
  size_t i = first;
  size_t j = last;
  double tmp = 0;
  double x =
      std::max(std::min(a[first], a[(first + last) / 2]),
               std::min(std::max(a[first], a[(first + last) / 2]),
                        a[last]));  // выбор опорного элемента как медианы первого, среднего и последнего элементов
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
  HoaraSort(a, i + 1, last);
  HoaraSort(a, first, j);
}

void deryabin_m_hoare_sort_simple_merge_stl::MergeTwoParts(std::vector<double>& arr, size_t left, size_t right) {
  std::inplace_merge(arr.begin() + left, arr.begin() + ((left + right) >> 1) + 1, arr.begin() + right + 1);
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
  const auto num_of_lvls = [](size_t n) {
    size_t log = 0;
    while (n >>= 1) ++log;
    return log;
  };
  for (size_t i = 0; i < num_of_lvls(chunk_count_); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      MergeTwoParts(input_array_A_, j * min_chunk_size_ << (i + 1), ((j + 1) * min_chunk_size_ << (i + 1)) - 1);
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
  const size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  auto parallel_for = [&](size_t start, size_t end, auto&& func) {
    const size_t num_chunk_per_thread = (end - start) / num_threads;
    for (size_t i = 0; i < num_threads - 1; ++i) {
      workers.emplace_back([=, &func] {
        for (size_t j = start + i * num_chunk_per_thread; j < start + (i + 1) * num_chunk_per_thread;) {
          func(j++);
        }
      });
    }
    workers.emplace_back([=, &func] {
      for (size_t j = start + (num_threads - 1) * num_chunk_per_thread; j < end;) {
        func(j++);
      }
    });
    for (auto& worker : workers) {
      worker.join();
    }
    workers.clear();
  };
  parallel_for(0, chunk_count_, [this](size_t count) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
  });
  const auto num_of_lvls = [](size_t n) {
    size_t log = 0;
    while (n >>= 1) ++log;
    return log;
  };
  for (size_t i = 0; i < num_of_lvls(chunk_count_); ++i) {
    parallel_for(0, chunk_count_ >> (i + 1), [this, i](size_t j) {
      MergeTwoParts(input_array_A_, j * min_chunk_size_ << (i + 1), ((j + 1) * min_chunk_size_ << (i + 1)) - 1);
    });
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
