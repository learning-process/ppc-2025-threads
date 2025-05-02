#include "stl/sozonov_i_image_filtering_block_partitioning/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {

std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

void ProcessBlockSet(const std::vector<double> &image, std::vector<double> &filtered_image, int width, int height,
                     int block_size, const std::vector<std::pair<int, int>> &blocks) {
  for (auto [start_i, start_j] : blocks) {
    int end_i = std::min(start_i + block_size, height - 1);
    int end_j = std::min(start_j + block_size, width - 1);
    for (int i = std::max(1, start_i); i < end_i; ++i) {
      for (int j = std::max(1, start_j); j < end_j; ++j) {
        double sum = 0;
        for (int l = -1; l <= 1; ++l) {
          for (int k = -1; k <= 1; ++k) {
            sum += image[((i - l) * width) + (j - k)] * kernel[((l + 1) * 3) + (k + 1)];
          }
        }
        filtered_image[(i * width) + j] = sum;
      }
    }
  }
}

}  // namespace

bool sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL::PreProcessingImpl() {
  // Init image
  image_ = std::vector<double>(task_data->inputs_count[0]);
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

  width_ = static_cast<int>(task_data->inputs_count[1]);
  height_ = static_cast<int>(task_data->inputs_count[2]);

  // Init filtered image
  filtered_image_ = std::vector<double>(width_ * height_, 0);
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL::ValidationImpl() {
  // Init image
  image_ = std::vector<double>(task_data->inputs_count[0]);
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

  size_t img_size = task_data->inputs_count[1] * task_data->inputs_count[2];

  // Check pixels range from 0 to 255
  for (size_t i = 0; i < img_size; ++i) {
    if (image_[i] < 0 || image_[i] > 255) {
      return false;
    }
  }

  // Check size of image
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == img_size &&
         task_data->outputs_count[0] == img_size && task_data->inputs_count[1] >= 3 && task_data->inputs_count[2] >= 3;
}

bool sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL::RunImpl() {
  int block_size = 64;
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  std::vector<std::vector<std::pair<int, int>>> thread_blocks(num_threads);

  int block_id = 0;
  for (int i = 0; i < height_; i += block_size) {
    for (int j = 0; j < width_; j += block_size) {
      thread_blocks[block_id % num_threads].emplace_back(i, j);
      block_id++;
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread(ProcessBlockSet, std::cref(image_), std::ref(filtered_image_), width_, height_, block_size,
                             thread_blocks[t]);
  }

  for (auto &t : threads) t.join();

  return true;
}

bool sozonov_i_image_filtering_block_partitioning_stl::TestTaskSTL::PostProcessingImpl() {
  auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  return true;
}
