#include "seq/varfolomeev_g_histogram_linear_stretching/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  img = std::vector<int>(in_ptr, in_ptr + input_size);
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  if (task_data->inputs_count[0] == 0 || task_data->outputs_count[0] == 0) {
    return false;
  }
  if (task_data->inputs_count[0] == task_data->outputs_count[0]) {
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    for (size_t i = 0; i < task_data->inputs_count[0]; ++i) {
      if (in_ptr[i] < 0 || in_ptr[i] > 255) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::RunImpl() {
  int min = 255;
  int max = 0;
  for (size_t i = 0; i < img.size(); i++) {
    int pixel = img[i];
    if (pixel < min) {
      min = pixel;
    }
    if (pixel > max) {
      max = pixel;
    }
  }

  if (max == min) {
    return true;
  }

  for (size_t i = 0; i < img.size(); i++) {
    img[i] = round(((img[i] - min) * 255.0) / (max - min));
  }

  return true;
}

bool varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < img.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = img[i];
  }
  return true;
}
