#include "all/komshina_d_image_filtering_vertical_gaussian/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];

  unsigned int input_size = width_ * height_ * 3;
  auto* in_ptr = reinterpret_cast<unsigned char*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int kernel_size = task_data->inputs_count[2];
  auto* kernel_ptr = reinterpret_cast<float*>(task_data->inputs[1]);
  kernel_.assign(kernel_ptr, kernel_ptr + kernel_size);

  output_.resize(input_.size(), 0);

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

  std::size_t rows_per_proc = height_ / size;
  std::size_t start_row = rank * rows_per_proc;
  std::size_t end_row = (rank == size - 1) ? height_ : start_row + rows_per_proc;

  std::vector<unsigned char> local_output = FilterLocalRegion(start_row, end_row);

  if (rank == 0) {
    output_.resize(input_.size());
    std::ranges::copy(local_output, output_.begin());

    for (int src = 1; src < size; ++src) {
      std::size_t src_start = src * rows_per_proc;
      std::size_t src_end = (src == size - 1) ? height_ : src_start + rows_per_proc;
      std::size_t chunk_size = (src_end - src_start) * width_ * 3;

      std::vector<unsigned char> temp(chunk_size);
      world_.recv(src, 0, temp);
      std::ranges::copy(temp, output_.begin() + static_cast<std::ptrdiff_t>(src_start * width_ * 3));
    }
  } else {
    world_.send(0, 0, local_output);
  }

  world_.barrier();
  return true;
}

std::vector<unsigned char> komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::FilterLocalRegion(
    std::size_t start_row, std::size_t end_row) const {
  const int kernel_radius = static_cast<int>(kernel_.size() / 2);
  std::size_t local_height = end_row - start_row;
  std::vector<unsigned char> local_output(local_height * width_ * 3);

  int start_row_int = static_cast<int>(start_row);
  int end_row_int = static_cast<int>(end_row);
  int width_int = static_cast<int>(width_);
  int height_int = static_cast<int>(height_);

#pragma omp parallel for default(none) shared(local_output) \
    firstprivate(start_row_int, end_row_int, width_int, height_int, kernel_radius)
  for (int y = start_row_int; y < end_row_int; ++y) {
    for (int x = 0; x < width_int; ++x) {
      for (int c = 0; c < 3; ++c) {
        float total = 0.0F;
        for (int k = -kernel_radius; k <= kernel_radius; ++k) {
          int yk = y + k;
          if (yk < 0 || yk >= height_int) {
            continue;
          }
          std::size_t idx = ((yk * width_int + x) * 3) + c;
          total += static_cast<float>(input_[idx]) * kernel_[k + kernel_radius];
        }
        std::size_t local_y = y - start_row_int;
        local_output[((local_y * width_int + x) * 3) + c] = std::clamp(static_cast<int>(std::round(total)), 0, 255);
      }
    }
  }

  return local_output;
}

bool komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, task_data->outputs[0]);
  }
  return true;
}
