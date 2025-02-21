#include "seq/malyshev_a_increase_contrast_by_histogram/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::PreProcessingImpl() {
  data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);

  return !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::RunImpl() {
  uint8_t min = 255;
  uint8_t max = 0;

  for (size_t i = 0; i < data_.size(); i++) {
    min = std::min(min, data_[i]);
    max = std::max(max, data_[i]);
  }

  if (min == max) {
    return true;
  }

  for (size_t i = 0; i < data_.size(); i++) {
    data_[i] = (data_[i] - min) * 255 / (max - min);
  }
  return true;
}

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < data_.size(); ++i) {
    task_data->outputs[0][i] = data_[i];
  }

  return true;
}
