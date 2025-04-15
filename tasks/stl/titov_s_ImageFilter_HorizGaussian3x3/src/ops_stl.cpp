#include "stl/titov_s_ImageFilter_HorizGaussian3x3/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <vector>

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::RunImpl() {
  const double sum = kernel_[0] + kernel_[1] + kernel_[2];
  const int width = width_;
  const int height = height_;

  std::vector<int> rows(height);
  std::iota(rows.begin(), rows.end(), 0);

  std::for_each(std::execution::par, rows.begin(), rows.end(),
                [=, &input = input_, &output = output_, &kernel = kernel_](int i) {
                  const int row_offset = i * width;
                  for (int j = 0; j < width; ++j) {
                    double filtered_value = input[row_offset + j] * kernel[1];
                    if (j > 0) {
                      filtered_value += input[row_offset + j - 1] * kernel[0];
                    }
                    if (j < width - 1) {
                      filtered_value += input[row_offset + j + 1] * kernel[2];
                    }
                    output[row_offset + j] = filtered_value / sum;
                  }
                });

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);

  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }

  return true;
}
