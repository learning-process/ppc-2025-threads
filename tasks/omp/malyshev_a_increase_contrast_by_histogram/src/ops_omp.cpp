#include "omp/malyshev_a_increase_contrast_by_histogram/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

bool malyshev_a_increase_contrast_by_histogram_omp::TestTaskOMP::PreProcessingImpl() {
  data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);

  return !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_omp::TestTaskOMP::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool malyshev_a_increase_contrast_by_histogram_omp::TestTaskOMP::RunImpl() {
  auto min_value = std::numeric_limits<uint8_t>::max();
  auto max_value = std::numeric_limits<uint8_t>::min();

  int data_size = static_cast<int>(data_.size());
#pragma omp parallel for reduction(min : min_value) reduction(max : max_value)
  for (int i = 0; i < data_size; i++) {
    min_value = std::min(min_value, data_[i]);
    max_value = std::max(max_value, data_[i]);
  }

  if (min_value == max_value) {
    return true;
  }

#pragma omp parallel for
  for (int i = 0; i < data_size; i++) {
    data_[i] = static_cast<uint8_t>((data_[i] - min_value) * 255 / (max_value - min_value));
  }

  return true;
}

bool malyshev_a_increase_contrast_by_histogram_omp::TestTaskOMP::PostProcessingImpl() {
  std::ranges::copy(data_, task_data->outputs[0]);

  return true;
}
