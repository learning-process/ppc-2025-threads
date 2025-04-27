#include "stl/malyshev_a_increase_contrast_by_histogram/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::PreProcessingImpl() {
  data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);
  return !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::RunImpl() {
  size_t data_size = data_.size();
  size_t num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  size_t grain_size = data_size / num_threads;

  std::vector<std::pair<uint8_t, uint8_t>> local_minmax(
      num_threads, std::make_pair(std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min()));

  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; ++i) {
    size_t start = i * grain_size;
    size_t end = (i == num_threads - 1) ? data_size : start + grain_size;
    threads.emplace_back([&, i, start, end]() {
      auto& local = local_minmax[i];
      for (size_t j = start; j < end; ++j) {
        local.first = std::min(local.first, data_[j]);
        local.second = std::max(local.second, data_[j]);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  uint8_t min_value = std::numeric_limits<uint8_t>::max();
  uint8_t max_value = std::numeric_limits<uint8_t>::min();
  for (const auto& local : local_minmax) {
    min_value = std::min(min_value, local.first);
    max_value = std::max(max_value, local.second);
  }

  if (min_value == max_value) {
    return true;
  }

  const auto spectrum = std::numeric_limits<uint8_t>::max();
  const auto range = max_value - min_value;

  threads.clear();
  for (size_t i = 0; i < num_threads; ++i) {
    size_t start = i * grain_size;
    size_t end = (i == num_threads - 1) ? data_size : start + grain_size;
    threads.emplace_back([&, start, end]() {
      for (size_t j = start; j < end; ++j) {
        data_[j] = static_cast<uint8_t>((data_[j] - min_value) * spectrum / range);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return true;
}

bool malyshev_a_increase_contrast_by_histogram_stl::TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(data_, task_data->outputs[0]);
  return true;
}
