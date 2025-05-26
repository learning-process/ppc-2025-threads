#include "omp/makhov_m_linear_image_filtering_vertical/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

bool makhov_m_linear_image_filtering_vertical_omp::TaskSequential::PreProcessingImpl() {
  // Init value for input, output, kernel
  width_ = (int)(task_data->inputs_count[0]);
  height_ = (int)(task_data->inputs_count[1]);
  input_size_ = width_ * height_ * 3;
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size_);
  output_.assign(input_.begin(), input_.end());
  kernel_ = {0.25F, 0.5F, 0.25F};  // [1, 2, 1] * 1/4
  return true;
}

bool makhov_m_linear_image_filtering_vertical_omp::TaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return ((task_data->inputs_count[0] * task_data->inputs_count[1] * 3 >= 27) &&
          ((task_data->inputs_count[0] * task_data->inputs_count[1] * 3) == task_data->outputs_count[0]));
}

bool makhov_m_linear_image_filtering_vertical_omp::TaskSequential::RunImpl() {
  std::vector<uint8_t> temp(input_size_);
  makhov_m_linear_image_filtering_vertical_omp::TaskSequential::ApplyVerticalGaussian(input_, temp, width_, height_,
                                                                                      kernel_);
  makhov_m_linear_image_filtering_vertical_omp::TaskSequential::ApplyHorizontalGaussian(temp, output_, width_, height_,
                                                                                        kernel_);
  return true;
}

bool makhov_m_linear_image_filtering_vertical_omp::TaskSequential::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], output_.data(),
              std::min(output_.size(), static_cast<size_t>(task_data->outputs_count[0])));
  return true;
}

template <typename AccessPixelFunc>
static void makhov_m_linear_image_filtering_vertical_omp::TaskSequential::ApplyGaussianImpl(
    const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height, const std::vector<float> &kernel,
    AccessPixelFunc get_pixel_index) {
  const int kernel_radius = static_cast<int>(kernel.size() / 2);
  const int channels = 3;

#pragma omp parallel for schedule(static)
  for (int outer = 0; outer < (get_pixel_index.is_horizontal ? height : width); ++outer) {
    for (int inner = 0; inner < (get_pixel_index.is_horizontal ? width : height); ++inner) {
      float sum_r = 0.0F, sum_g = 0.0F, sum_b = 0.0F;

      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int idx = get_pixel_index(inner, outer, k, width, height);
        float weight = kernel[k + kernel_radius];

        sum_r += static_cast<float>(src[idx]) * weight;
        sum_g += static_cast<float>(src[idx + 1]) * weight;
        sum_b += static_cast<float>(src[idx + 2]) * weight;
      }

      int dst_idx = ((get_pixel_index.is_horizontal ? outer * width + inner : inner * width + outer) * channels);
      dst[dst_idx] = static_cast<uint8_t>(std::clamp(sum_r, 0.0F, 255.0F));
      dst[dst_idx + 1] = static_cast<uint8_t>(std::clamp(sum_g, 0.0F, 255.0F));
      dst[dst_idx + 2] = static_cast<uint8_t>(std::clamp(sum_b, 0.0F, 255.0F));
    }
  }
}

// Applying 1D Gaussian Kernel to a row (RGB version)
void makhov_m_linear_image_filtering_vertical_omp::TaskSequential::ApplyHorizontalGaussian(
    const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height,
    const std::vector<float> &kernel) {
  struct {
    bool is_horizontal = true;
    int operator()(int x, int y, int k, int width, int) const {
      int pixel_x = std::clamp(x + k, 0, width - 1);
      return (y * width + pixel_x) * 3;
    }
  } horizontal_accessor;

  ApplyGaussianImpl(src, dst, width, height, kernel, horizontal_accessor);
}

// Applying 1D Gaussian Kernel to a column (RGB version)
void makhov_m_linear_image_filtering_vertical_omp::TaskSequential::ApplyVerticalGaussian(
    const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height,
    const std::vector<float> &kernel) {
  struct {
    bool is_horizontal = false;
    int operator()(int y, int x, int k, int width, int height) const {
      int pixel_y = std::clamp(y + k, 0, height - 1);
      return (pixel_y * width + x) * 3;
    }
  } vertical_accessor;

  ApplyGaussianImpl(src, dst, width, height, kernel, vertical_accessor);
}
