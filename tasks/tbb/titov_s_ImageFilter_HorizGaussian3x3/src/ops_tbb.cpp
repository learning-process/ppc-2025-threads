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
  const double k0 = static_cast<double>(kernel_[0]);
  const double k1 = static_cast<double>(kernel_[1]);
  const double k2 = static_cast<double>(kernel_[2]);
  const double sum = k0 + k1 + k2;

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());

  arena.execute([&] {
    tbb::task_group tg;

    const int threads_num = ppc::util::GetPPCNumThreads();
    const int rows_per_thread = height_ / threads_num;
    const int remainder_rows = height_ % threads_num;

    int start_row = 0;
    for (int i = 0; i < threads_num; ++i) {
      const int end_row = start_row + rows_per_thread + (i < remainder_rows ? 1 : 0);

      if (start_row < end_row) {
        tg.run([=, &input_ = input_, &output_ = output_] {
          for (int row = start_row; row < end_row; ++row) {
            for (int col = 0; col < width_; ++col) {
              double left_val = (col > 0) ? input_[row * width_ + (col - 1)] : 0.0;

              double center_val = input_[row * width_ + col];

              double right_val = (col < width_ - 1) ? input_[row * width_ + (col + 1)] : 0.0;

              double val = left_val * k0 + center_val * k1 + right_val * k2;
              output_[row * width_ + col] = val / sum;
            }
          }
        });
      }
      start_row = end_row;
    }

    tg.wait();
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
