#include "tbb/titov_s_ImageFilter_HorizGaussian3x3/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::RunImpl() {
  const double k0 = kernel_[0], k1 = kernel_[1], k2 = kernel_[2];
  const double sum = k0 + k1 + k2;

  oneapi::tbb::parallel_for(0, height_, [&](int row) {
    for (int col = 0; col < width_; ++col) {
      double left = (col > 0) ? input_[row * width_ + col - 1] : 0.0;
      double center = input_[row * width_ + col];
      double right = (col < width_ - 1) ? input_[row * width_ + col + 1] : 0.0;

      output_[row * width_ + col] = (left * k0 + center * k1 + right * k2) / sum;
    }
  });

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_tbb::ImageFilterTBB::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);

  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }

  return true;
}
