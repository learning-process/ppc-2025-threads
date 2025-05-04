#include "stl/golovkin_contrast_stretching/include/ops_stl.hpp"

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>

bool golovkin_contrast_stretching::ContrastStretchingSTL::ValidationImpl() {
  // Ожидаем один вход (одномерный массив), один выход того же размера
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool golovkin_contrast_stretching::ContrastStretchingSTL::PreProcessingImpl() {
  image_size_ = task_data->inputs_count[0];

  auto* input_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
  input_image_.assign(input_ptr, input_ptr + image_size_);
  output_image_.resize(image_size_);

  // Поиск min и max
  auto [min_it, max_it] = std::minmax_element(input_image_.begin(), input_image_.end());
  min_val_ = *min_it;
  max_val_ = *max_it;

  return true;
}

bool golovkin_contrast_stretching::ContrastStretchingSTL::RunImpl() {
  if (min_val_ == max_val_) {
    std::fill(output_image_.begin(), output_image_.end(), 0);
    return true;
  }

  std::transform(input_image_.begin(), input_image_.end(), output_image_.begin(), [this](uint8_t pixel) -> uint8_t {
    return static_cast<uint8_t>((static_cast<int>(pixel) - min_val_) * 255 / (max_val_ - min_val_));
  });

  return true;
}

bool golovkin_contrast_stretching::ContrastStretchingSTL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<uint8_t*>(task_data->outputs[0]);
  std::copy(output_image_.begin(), output_image_.end(), output_ptr);
  return true;
}