#include "tbb/komshina_d_image_filtering_vertical_gaussian/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

  unsigned int input_size = width_ * height_ * 3;
  auto *in_ptr = reinterpret_cast<unsigned char *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int kernel_size = task_data->inputs_count[2];
  auto *kernel_ptr = reinterpret_cast<float *>(task_data->inputs[1]);
  kernel_.assign(kernel_ptr, kernel_ptr + kernel_size);

  output_.assign(input_.begin(), input_.end());

  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr || task_data->outputs.empty() ||
      task_data->outputs[0] == nullptr) {
    return false;
  }

  const auto &input_count = task_data->inputs_count;
  const auto &output_count = task_data->outputs_count;

  if (input_count.size() < 3 || output_count.empty()) {
    return false;
  }

  constexpr int kKernelSize = 9;
  constexpr int kChannels = 3;

  bool valid_kernel = (input_count[2] == kKernelSize);
  bool valid_size = (input_count[0] * input_count[1] * kChannels == output_count[0]);

  return valid_kernel && valid_size;
}

void VerticalGaussianFilter(const std::vector<unsigned char> &input, int width, int height,
                            const std::vector<float> &kernel, std::vector<unsigned char> &output) {
  const int kernel_radius = 1;
  const int channels = 3;


  tbb::parallel_for(1, height - 1, [&](int y) {
    for (int x = 1; x + 1 < width; ++x) {
      std::size_t base_idx = (y * width + x) * channels;

      for (int c = 0; c < channels; ++c) {
        float total = 0.0F;
        std::size_t k_idx = 0;

        for (int ky = -1; ky <= 1; ++ky) {
          std::size_t row_idx = (((y + ky) * width + (x - 1)) * channels) + c;

          for (int kx = -1; kx <= 1; ++kx, ++k_idx) {
            total += static_cast<float>(input[row_idx]) * kernel[k_idx];
            row_idx += channels;
          }
        }

        output[base_idx + c] = std::clamp(static_cast<int>(std::round(total)), 0, 255);
      }
    }
  });
}

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::RunImpl() {
  oneapi::tbb::task_arena arena(1);
  arena.execute([&] {
    tbb::task_group tg;
    for (int thr = 0; thr < ppc::util::GetPPCNumThreads(); ++thr) {
      tg.run([&] { VerticalGaussianFilter(input_, width_, height_, kernel_, output_); });
    }
    tg.wait();
  });
  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_tbb::TestTaskTBB::PostProcessingImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto *output_ptr = task_data->outputs[0];
  std::ranges::copy(output_, output_ptr);

  return true;
}
