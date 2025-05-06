#include "all/frolova_e_Sobel_filter/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

std::vector<int> frolova_e_sobel_filter_all::ToGrayScaleImg(std::vector<frolova_e_sobel_filter_all::RGB>& color_img,
                                                            size_t width, size_t height) {
  std::vector<int> gray_scale_image(width * height);
  for (size_t i = 0; i < width * height; i++) {
    gray_scale_image[i] =
        static_cast<int>((0.299 * color_img[i].R) + (0.587 * color_img[i].G) + (0.114 * color_img[i].B));
  }

  return gray_scale_image;
}

int frolova_e_sobel_filter_all::GetPixelSafe(const std::vector<int>& img, size_t x, size_t y, size_t width,
                                             size_t height) {
  if (x >= width || y >= height) {
    return 0;
  }
  return img[(y * width) + x];
}

bool frolova_e_sobel_filter_all::SobelFilterALL::PreProcessingImpl() {
  // Init value for input and output
  if (world.rank() == 0) {
    int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);
    width_ = static_cast<size_t>(value_1[0]);
    height_ = static_cast<size_t>(value_1[1]);

    int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
    std::vector<int> picture_vector;
    picture_vector.assign(value_2, value_2 + task_data->inputs_count[1]);

    for (size_t i = 0; i < picture_vector.size(); i += 3) {
      RGB pixel;
      pixel.R = picture_vector[i];
      pixel.G = picture_vector[i + 1];
      pixel.B = picture_vector[i + 2];
      picture_.push_back(pixel);
    }
    grayscale_image_ = frolova_e_sobel_filter_all::ToGrayScaleImg(picture_, width_, height_);

    res_image_.resize(width_ * height_);
  }

  return true;
}

bool frolova_e_sobel_filter_all::SobelFilterALL::ValidationImpl() {
  // Check equality of counts elements
  if (world.rank() == 0) {
    int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);

    if (task_data->inputs_count[0] != 2) {
      return false;
    }

    if (value_1[0] <= 0 || value_1[1] <= 0) {
      return false;
    }

    auto width_1 = static_cast<size_t>(value_1[0]);
    auto height_1 = static_cast<size_t>(value_1[1]);

    int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
    std::vector<int> picture_vector(value_2, value_2 + task_data->inputs_count[1]);
    if (task_data->inputs_count[1] != width_1 * height_1 * 3) {
      return false;
    }

    for (size_t i = 0; i < picture_vector.size(); i++) {
      if (picture_vector[i] < 0 || picture_vector[i] > 255) {
        return false;
      }
    }
  }
  return true;
}

bool frolova_e_sobel_filter_all::SobelFilterALL::RunImpl() {
  int rank = world.rank();
  int size = world.size();

  broadcast(world, height_, 0);
  broadcast(world, width_, 0);

  int active_processes_ = std::min(size, static_cast<int>(height_));

  if (rank < active_processes_) {
    int rows_per_proc = height_ / active_processes_;
    int remainder = height_ % active_processes_;

    int y_start = rank * rows_per_proc + std::min(rank, remainder);
    int y_end = y_start + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_rows = y_end - y_start;

    int has_top = (rank > 0) ? 1 : 0;
    int has_bottom = (rank < active_processes_ - 1) ? 1 : 0;
    int extended_rows = local_rows + has_top + has_bottom;

    local_image.resize(extended_rows * width_);
    std::vector<int> local_result(local_rows * width_);

    if (rank == 0) {
      std::vector<int> gray = grayscale_image_;
      for (int proc = 0; proc < active_processes_; proc++) {
        int proc_y_start = proc * rows_per_proc + std::min(proc, remainder);
        int proc_y_end = proc_y_start + rows_per_proc + (proc < remainder ? 1 : 0);
        int proc_local_rows = proc_y_end - proc_y_start;

        int top = (proc > 0) ? 1 : 0;
        int bottom = (proc < active_processes_ - 1) ? 1 : 0;
        int ext_rows = proc_local_rows + top + bottom;

        std::vector<int> chunk(ext_rows * width_, 0);
        for (int i = 0; i < ext_rows; ++i) {
          int src_y = proc_y_start - top + i;
          if (src_y >= 0 && src_y < static_cast<int>(height_)) {
            std::copy_n(gray.begin() + src_y * width_, width_, chunk.begin() + i * width_);
          } else {
            std::fill_n(chunk.begin() + i * width_, width_, 0);
          }
        }

        if (proc == 0) {
          local_image = chunk;
        } else {
          world.send(proc, 0, chunk);
        }
      }
    } else {
      world.recv(0, 0, local_image);
    }

    tbb::parallel_for(
        tbb::blocked_range<int>(has_top, local_rows + has_top), [&](const tbb::blocked_range<int>& range) {
          for (int y = range.begin(); y < range.end(); ++y) {
            for (size_t x = 0; x < width_; ++x) {
              int gx = 0;
              int gy = 0;

              for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                  int pixel =
                      frolova_e_sobel_filter_all::GetPixelSafe(local_image, x + dx, y + dy, width_, extended_rows);

                  int kx = (dx == -1 ? -1 : (dx == 1 ? 1 : 0)) * (dy == 0 ? 2 : 1);
                  int ky = (dy == -1 ? -1 : (dy == 1 ? 1 : 0)) * (dx == 0 ? 2 : 1);

                  gx += pixel * kx;
                  gy += pixel * ky;
                }
              }

              int val = std::sqrt(gx * gx + gy * gy);
              val = std::clamp(val, 0, 255);
              local_result[(y - has_top) * width_ + x] = val;
            }
          }
        });

    if (world.rank() == 0) {
      std::copy(local_result.begin(), local_result.end(), res_image_.begin() + y_start * width_);

      for (int proc = 1; proc < active_processes_; ++proc) {
        int proc_y_start = proc * rows_per_proc + std::min(proc, remainder);
        int proc_y_end = proc_y_start + rows_per_proc + (proc < remainder ? 1 : 0);
        int proc_local_rows = proc_y_end - proc_y_start;

        std::vector<int> proc_result(proc_local_rows * width_);

        world.recv(proc, 1, proc_result);

        std::copy(proc_result.begin(), proc_result.end(), res_image_.begin() + proc_y_start * width_);
      }
    } else {
      world.send(0, 1, local_result);
    }
  }

  return true;
}

bool frolova_e_sobel_filter_all::SobelFilterALL::PostProcessingImpl() {
  if (world.rank() == 0) {
    for (size_t i = 0; i < width_ * height_; i++) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = res_image_[i];
    }
  }

  return true;
}
