#include "tbb/titov_s_ImageFilter_HorizGaussian3x3/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

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
  const auto k0 = static_cast<double>(kernel_[0]);
  const auto k1 = static_cast<double>(kernel_[1]);
  const auto k2 = static_cast<double>(kernel_[2]);
  const double inv_sum = 1.0 / (k0 + k1 + k2);

  const int threads_num = ppc::util::GetPPCNumThreads();
  const int min_block_size = 512;
  const int total_rows = height_;
  const int base_chunk = std::max(min_block_size, total_rows / threads_num);

  oneapi::tbb::task_arena arena(threads_num);
  arena.execute([&] {
    tbb::parallel_for(tbb::blocked_range<int>(0, height_, base_chunk), [&](const tbb::blocked_range<int> &range) {
      for (int row = range.begin(); row < range.end(); ++row) {
        for (int col = 0; col < width_; ++col) {
          const double left = (col > 0) ? input_[row * width_ + col - 1] : 0.0;
          const double center = input_[row * width_ + col];
          const double right = (col < width_ - 1) ? input_[row * width_ + col + 1] : 0.0;
          output_[row * width_ + col] = (left * k0 + center * k1 + right * k2) * inv_sum;
        }
      }
    });
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
