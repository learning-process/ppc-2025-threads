#include "all/komshina_d_image_filtering_vertical_gaussian/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace {

void VerticalGaussianFilter(const std::vector<unsigned char>& local_input, std::vector<unsigned char>& local_output,
                            const std::vector<float>& kernel, std::size_t local_height, std::size_t width,
                            std::size_t k_radius, std::size_t halo_top) {
  const std::size_t k_channels = 3;

#pragma omp parallel for default(none) \
    shared(local_input, local_output, kernel, local_height, width, k_radius, halo_top)
  for (std::ptrdiff_t y = static_cast<std::ptrdiff_t>(halo_top);
       y < static_cast<std::ptrdiff_t>(halo_top + local_height); ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      std::size_t base_idx = ((static_cast<std::size_t>(y) - halo_top) * width + x) * k_channels;

      for (std::size_t c = 0; c < k_channels; ++c) {
        float sum = 0.0f;

        for (std::ptrdiff_t k = -static_cast<std::ptrdiff_t>(k_radius); k <= static_cast<std::ptrdiff_t>(k_radius);
             ++k) {
          std::ptrdiff_t yy = y + k;
          if (yy < 0 || yy >= static_cast<std::ptrdiff_t>(local_height)) continue;

          std::size_t idx = (static_cast<std::size_t>(yy) * width + x) * k_channels + c;
          sum += static_cast<float>(local_input[idx]) * kernel[k + k_radius];
        }

        local_output[base_idx + c] = std::clamp(static_cast<int>(std::round(sum)), 0, 255);
      }
    }
  }
}

}  // namespace

bool komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

  unsigned int input_size = width_ * height_ * 3;
  auto* in_ptr = reinterpret_cast<unsigned char*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int kernel_size = task_data->inputs_count[2];
  auto* kernel_ptr = reinterpret_cast<float*>(task_data->inputs[1]);
  kernel_.assign(kernel_ptr, kernel_ptr + kernel_size);

  output_.resize(input_size, 0);
  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->inputs[1] == nullptr || task_data->outputs.empty() ||
      task_data->outputs[0] == nullptr) {
    return false;
  }

  const auto& input_count = task_data->inputs_count;
  const auto& output_count = task_data->outputs_count;

  if (input_count.size() < 3 || output_count.empty()) {
    return false;
  }

  constexpr int kKernelSize = 9;
  constexpr int kChannels = 3;

  bool valid_kernel = (input_count[2] == kKernelSize);
  bool valid_size = (input_count[0] * input_count[1] * kChannels == output_count[0]);

  return valid_kernel && valid_size;
}

bool komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::RunImpl() {
  const int rank = world_.rank();
  const int size = world_.size();
  const std::size_t k_channels = 3;
  const std::size_t k_radius = kernel_.size() / 2;

  std::size_t rows_per_proc = height_ / static_cast<std::size_t>(size);
  std::size_t remainder = height_ % static_cast<std::size_t>(size);
  std::size_t local_height = rows_per_proc + (rank < remainder ? 1 : 0);
  std::size_t offset = (rank * rows_per_proc) + std::min<std::size_t>(rank, remainder);

  std::size_t halo_top = (offset > 0) ? k_radius : 0;
  std::size_t halo_bottom = ((offset + local_height) < height_) ? k_radius : 0;
  std::size_t total_rows = local_height + halo_top + halo_bottom;
  std::size_t local_input_size = total_rows * width_ * k_channels;

  std::vector<unsigned char> local_input(local_input_size);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      std::size_t lh = rows_per_proc + (i < remainder ? 1 : 0);
      std::size_t off = (i * rows_per_proc) + std::min<std::size_t>(i, remainder);

      std::size_t halo_t = (off > 0) ? k_radius : 0;
      std::size_t halo_b = ((off + lh) < height_) ? k_radius : 0;
      std::size_t rows = lh + halo_t + halo_b;

      std::ptrdiff_t start_row = static_cast<std::ptrdiff_t>(off) - static_cast<std::ptrdiff_t>(halo_t);
      std::size_t start_idx = std::max<ptrdiff_t>(start_row, 0) * width_ * k_channels;
      std::size_t count = rows * width_ * k_channels;

      if (i == 0) {
        std::copy(input_.begin() + start_idx, input_.begin() + start_idx + count, local_input.begin());
      } else {
        std::vector<unsigned char> temp(input_.begin() + start_idx, input_.begin() + start_idx + count);
        world_.send(i, 0, temp);
      }
    }
  } else {
    world_.recv(0, 0, local_input);
  }

  std::vector<unsigned char> local_output(local_height * width_ * k_channels, 0);

  VerticalGaussianFilter(local_input, local_output, kernel_, local_height, width_, k_radius, halo_top);

  if (rank == 0) {
    std::ranges::copy(local_output, output_.begin() + offset * width_ * k_channels);

    for (int i = 1; i < size; ++i) {
      std::size_t lh = rows_per_proc + (i < remainder ? 1 : 0);
      std::vector<unsigned char> temp(lh * width_ * k_channels);
      world_.recv(i, 1, temp);

      std::size_t target_offset = (i * rows_per_proc + std::min<std::size_t>(i, remainder)) * width_ * k_channels;
      std::ranges::copy(temp, output_.begin() + target_offset);
    }
  } else {
    world_.send(0, 1, local_output);
  }

  return true;
}

bool komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, reinterpret_cast<unsigned char*>(task_data->outputs[0]));
  }
  return true;
}