#include "seq/varfolomeev_g_histogram_linear_stretching/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  img_ = std::vector<int>(in_ptr, in_ptr + input_size);
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  if (task_data->inputs_count[0] == 0 || task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; ++i) {
    if (in_ptr[i] < 0 || in_ptr[i] > 255) {
      return false;
    }
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::RunImpl() {
  if (img_.size() < 1) {
    return false;
  }
  int min = *std::ranges::min_element(img_);
  int max = *std::ranges::max_element(img_);

  if (max != min) {
    for (size_t i = 0; i < img_.size(); i++) {
      img_[i] = static_cast<int>(round(((img_[i] - min) / static_cast<double>(max - min)) * 255.0));
    }
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < img_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = img_[i];
  }
  return true;
}
