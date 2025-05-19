#include "all/sozonov_i_image_filtering_block_partitioning/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
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
  if (world_.rank() == 0) {
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
  broadcast(world_, width_, 0);
  broadcast(world_, height_, 0);

  std::vector<int> block_size(world_.size());

  int delta = width_ / world_.size();
  if (width_ % world_.size() != 0) {
    delta++;
  }
  if (world_.rank() >= world_.size() - (world_.size() * delta) + width_) {
    delta--;
  }

  delta = delta + 2;

  gather(world_, delta, block_size.data(), 0);

  std::vector<double> local_image(delta * height_, 0);
  std::vector<double> send_image(delta * height_);

  if (world_.size() == 1) {
#pragma omp parallel for
    for (int i = 0; i < height_; ++i) {
      for (int j = 1; j < width_ + 1; ++j) {
        local_image[(i * (width_ + 2)) + j] = image_[(i * width_) + j - 1];
      }
    }
  } else {
    if (world_.rank() == 0) {
#pragma omp parallel for
      for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < delta - 1; ++j) {
          local_image[(i * delta) + j + 1] = image_[(i * width_) + j];
        }
      }
      int idx = delta - 2;
      for (int proc = 1; proc < world_.size(); ++proc) {
        send_image = std::vector<double>(delta * height_, 0);
        if (proc == world_.size() - 1) {
#pragma omp parallel for
          for (int i = 0; i < height_; ++i) {
            for (int j = -1; j < block_size[proc] - 2; ++j) {
              send_image[(i * block_size[proc]) + j + 1] = image_[(i * width_) + j + idx];
            }
          }
        } else {
#pragma omp parallel for
          for (int i = 0; i < height_; ++i) {
            for (int j = -1; j < block_size[proc] - 1; ++j) {
              send_image[(i * block_size[proc]) + j + 1] = image_[(i * width_) + j + idx];
            }
          }
          idx += block_size[proc] - 2;
        }
        world_.send(proc, 0, send_image.data(), block_size[proc] * height_);
      }
    } else {
      world_.recv(0, 0, local_image.data(), delta * height_);
    }
  }

  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

  std::vector<double> local_filtered_image(delta * height_, 0);

#pragma omp parallel for
  for (int i = 1; i < height_ - 1; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += local_image[((i - l) * delta) + j - k] * kernel[((l + 1) * 3) + k + 1];
        }
      }
      local_filtered_image[(i * delta) + j] = sum;
    }
  }

  std::vector<double> back_image((delta - 2) * height_);

#pragma omp parallel for
  for (int i = 0; i < height_; ++i) {
    for (int j = 1; j < delta - 1; ++j) {
      back_image[(i * (delta - 2)) + j - 1] = local_filtered_image[(i * delta) + j];
    }
  }

  std::vector<int> recv_counts(world_.size());

  if (world_.rank() == 0) {
#pragma omp parallel for
    for (int i = 0; i < world_.size(); ++i) {
      recv_counts[i] = (block_size[i] - 2) * height_;
    }
  }

  std::vector<double> gathered_image(width_ * height_);
  gatherv(world_, back_image, gathered_image.data(), recv_counts, 0);

  if (world_.rank() == 0) {
    int idx = 0;
    for (int proc = 0; proc < world_.size(); ++proc) {
#pragma omp parallel for
      for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < block_size[proc] - 2; ++j) {
          filtered_image_[(i * width_) + j + idx] = gathered_image[(i * (block_size[proc] - 2)) + j + (idx * height_)];
        }
      }
      idx += block_size[proc] - 2;
    }
  }

  return true;
}

bool sozonov_i_image_filtering_block_partitioning_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  }
  return true;
}
