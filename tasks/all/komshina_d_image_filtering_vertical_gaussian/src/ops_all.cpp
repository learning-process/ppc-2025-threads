#include "all/komshina_d_image_filtering_vertical_gaussian/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <numeric>

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
  const int kChannels = 3;
  const int kRadius = static_cast<int>(kernel_.size() / 2);

  int rows_per_proc = height_ / size;
  int remainder = height_ % size;
  int local_height = rows_per_proc + (rank < remainder ? 1 : 0);
  int offset = rank * rows_per_proc + std::min(rank, remainder);

  int halo_top = (offset > 0) ? kRadius : 0;
  int halo_bottom = ((offset + local_height) < static_cast<int>(height_)) ? kRadius : 0;
  int total_rows = local_height + halo_top + halo_bottom;
  int local_input_size = total_rows * width_ * kChannels;

  std::vector<unsigned char> local_input(local_input_size);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      int lh = rows_per_proc + (i < remainder ? 1 : 0);
      int off = i * rows_per_proc + std::min(i, remainder);

      int halo_t = (off > 0) ? kRadius : 0;
      int halo_b = ((off + lh) < static_cast<int>(height_)) ? kRadius : 0;
      int rows = lh + halo_t + halo_b;

      int start_row = off - halo_t;
      int start_idx = std::max(start_row, 0) * width_ * kChannels;
      int count = rows * width_ * kChannels;

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

  std::vector<unsigned char> local_output(local_height * width_ * kChannels, 0);

const std::size_t width = width_;
  const std::vector<float>& kernel = kernel_;

#pragma omp parallel for default(none) shared(local_input, local_output, kernel) \
    firstprivate(local_height, width, kRadius, halo_top)
  for (int y = halo_top; y < halo_top + local_height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      std::size_t base_idx = ((y - halo_top) * width + x) * kChannels;

      for (int c = 0; c < kChannels; ++c) {
        float sum = 0.0f;

        for (int k = -kRadius; k <= kRadius; ++k) {
          int yy = y + k;
          if (yy < 0 || yy >= static_cast<int>(total_rows)) continue;

          std::size_t idx = ((yy * width + x) * kChannels) + c;
          sum += static_cast<float>(local_input[idx]) * kernel[k + kRadius];
        }

        local_output[base_idx + c] = std::clamp(static_cast<int>(std::round(sum)), 0, 255);
      }
    }
  }

  if (rank == 0) {
    std::copy(local_output.begin(), local_output.end(), output_.begin() + offset * width_ * kChannels);

    for (int i = 1; i < size; ++i) {
      int lh = rows_per_proc + (i < remainder ? 1 : 0);
      std::vector<unsigned char> temp(lh * width_ * kChannels);
      world_.recv(i, 1, temp);

      int target_offset = (i * rows_per_proc + std::min(i, remainder)) * width_ * kChannels;
      std::copy(temp.begin(), temp.end(), output_.begin() + target_offset);
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