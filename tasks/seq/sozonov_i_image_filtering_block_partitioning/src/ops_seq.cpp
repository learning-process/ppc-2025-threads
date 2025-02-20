#include "seq/sozonov_i_image_filtering_block_partitioning/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::PreProcessingImpl() {
  // Init image
  unsigned int image_size = task_data->inputs_count[0];
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  image = std::vector<double>(image_ptr, image_ptr + image_size);

  width = task_data->inputs_count[1];
  height = task_data->inputs_count[2];

  // Init filtered image
  filtered_image = std::vector<double>(width * height, 0);
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::ValidationImpl() {
  // Init image
  unsigned int image_size = task_data->inputs_count[0];
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  image = std::vector<double>(image_ptr, image_ptr + image_size);

  size_t img_size = task_data->inputs_count[1] * task_data->inputs_count[2];

  // Check pixels range from 0 to 255
  for (size_t i = 0; i < img_size; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }

  // Check size of image
  return task_data->inputs_count[0] == img_size && task_data->outputs_count[0] == img_size &&
         task_data->inputs_count[1] >= 3 && task_data->inputs_count[2] >= 3;
}

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::RunImpl() {
  // Linear image filtering
  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += image[(i - l) * width + j - k] * kernel[(l + 1) * 3 + k + 1];
        }
      }
      filtered_image[i * width + j] = sum;
    }
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::PostProcessingImpl() {
  auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
  std::copy(filtered_image.begin(), filtered_image.end(), out);
  return true;
}
