#include "seq/tarakanov_d_contrast_enhancement_by_linear_histogram_stretching/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

bool tarakanov_d_linear_stretching::TaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<unsigned char *>(task_data->inputs[0]);

  rc_size_ = static_cast<int>(std::sqrt(input_size));

  inputImage_.resize(input_size);
  std::memcpy(inputImage_.data(), in_ptr, input_size * sizeof(unsigned char));

  outputImage_.resize(input_size, 0);

  return true;
}

bool tarakanov_d_linear_stretching::TaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool tarakanov_d_linear_stretching::TaskSequential::RunImpl() {
  unsigned char min_val = 255;
  unsigned char max_val = 0;
  size_t total_pixels = inputImage_.size();

  for (size_t idx = 0; idx < total_pixels; ++idx) {
    unsigned char pixel = inputImage_[idx];
    min_val = std::min(pixel, min_val);
    max_val = std::max(pixel, max_val);
  }

  if (min_val == max_val) {
    outputImage_ = inputImage_;
    return true;
  }

  for (size_t idx = 0; idx < total_pixels; ++idx) {
    unsigned char pixel = inputImage_[idx];
    auto new_pixel = static_cast<unsigned char>((pixel - min_val) * 255.0 / (max_val - min_val));
    outputImage_[idx] = new_pixel;
  }

  return true;
}

bool tarakanov_d_linear_stretching::TaskSequential::PostProcessingImpl() {
  size_t total_elements = outputImage_.size();
  auto *out_ptr = reinterpret_cast<unsigned char *>(task_data->outputs[0]);

  std::memcpy(out_ptr, outputImage_.data(), total_elements * sizeof(unsigned char));

  return true;
}
