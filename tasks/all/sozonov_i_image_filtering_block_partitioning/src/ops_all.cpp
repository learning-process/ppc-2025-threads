#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

void FilterBlock(const std::vector<double> &input, std::vector<double> &output, int ext_width, int ext_height) {
#pragma omp parallel for
  for (int i = 1; i < ext_height - 1; ++i) {
    for (int j = 1; j < ext_width - 1; ++j) {
      double sum = 0.0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += input[(i + l) * ext_width + (j + k)] * kernel[(l + 1) * 3 + (k + 1)];
        }
      }
      output[i * ext_width + j] = sum;
    }
  }
}

}  // namespace

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PreProcessingImpl() {
  if (world.rank() == 0) {
    // Init image
    image_ = std::vector<double>(task_data->inputs_count[0]);
    auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

    width_ = static_cast<int>(task_data->inputs_count[1]);
    height_ = static_cast<int>(task_data->inputs_count[2]);

    // Init filtered image
    filtered_image_ = std::vector<double>(width_ * height_, 0);
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::ValidationImpl() {
  if (world.rank() == 0) {
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
           task_data->outputs_count[0] == img_size && task_data->inputs_count[1] >= 3 &&
           task_data->inputs_count[2] >= 3;
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::RunImpl() {
  boost::mpi::broadcast(world, width_, 0);
  boost::mpi::broadcast(world, height_, 0);

  int rank = world.rank();
  int size = world.size();

  int blocks_x = static_cast<int>(std::sqrt(size));
  int blocks_y = size / blocks_x;

  int block_width = width_ / blocks_x;
  int block_height = height_ / blocks_y;

  int ext_width = block_width + 2;
  int ext_height = block_height + 2;

  int padded_block_size = ext_width * ext_height;
  int filtered_block_size = block_width * block_height;

  std::vector<std::vector<double>> padded_blocks;
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    padded_blocks.resize(size);

    for (int p = 0; p < size; ++p) {
      padded_blocks[p].resize(padded_block_size, 0.0);

      int bx = p % blocks_x;
      int by = p / blocks_x;
      int si = by * block_height;
      int sj = bx * block_width;

      for (int i = 0; i < ext_height; ++i) {
        for (int j = 0; j < ext_width; ++j) {
          int gi = si + i - 1;
          int gj = sj + j - 1;
          if (gi >= 0 && gi < height_ && gj >= 0 && gj < width_) {
            padded_blocks[p][i * ext_width + j] = image_[gi * width_ + gj];
          }
        }
      }

      sendcounts[p] = padded_block_size;
      displs[p] = p * padded_block_size;
    }
  }

  std::vector<double> all_data;
  if (rank == 0) {
    all_data.resize(size * padded_block_size);
    for (int p = 0; p < size; ++p) {
      std::copy(padded_blocks[p].begin(), padded_blocks[p].end(), all_data.begin() + p * padded_block_size);
    }
  }

  std::vector<double> local_input(padded_block_size);

  boost::mpi::scatterv(world, all_data, sendcounts, displs, local_input, 0);

  std::vector<double> local_output(padded_block_size, 0.0);
  FilterBlock(local_input, local_output, ext_width, ext_height);

  std::vector<double> local_filtered(filtered_block_size);
  for (int i = 0; i < block_height; ++i) {
    std::copy_n(&local_output[(i + 1) * ext_width + 1], block_width, &local_filtered[i * block_width]);
  }

  std::vector<double> gathered_data;
  if (rank == 0) gathered_data.resize(size * filtered_block_size);

  std::vector<int> recvcounts(size, filtered_block_size);
  std::vector<int> recvdispls(size);
  for (int p = 0; p < size; ++p) recvdispls[p] = p * filtered_block_size;

  boost::mpi::gatherv(world, local_filtered, gathered_data, recvcounts, recvdispls, 0);

  if (rank == 0) {
    filtered_image_.resize(width_ * height_);

    for (int p = 0; p < size; ++p) {
      int bx = p % blocks_x;
      int by = p / blocks_x;
      int si = by * block_height;
      int sj = bx * block_width;

      for (int i = 0; i < block_height; ++i) {
        std::copy_n(&gathered_data[p * filtered_block_size + i * block_width], block_width,
                    &filtered_image_[(si + i) * width_ + sj]);
      }
    }
  }

  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PostProcessingImpl() {
  if (world.rank() == 0) {
    auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  }
  return true;
}
