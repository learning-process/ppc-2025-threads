#include "stl/frolova_e_Sobel_filter/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

int frolova_e_sobel_filter_stl::GetPixelSafe(const std::vector<int>& img, size_t x, size_t y, size_t width,
                                             size_t height) {
  if (x >= width || y >= height) {
    return 0;
  }
  return img[(y * width) + x];
}

bool frolova_e_sobel_filter_stl::SobelFilterSTL::PreProcessingImpl() {
  // Init value for input and output
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

  res_image_.resize(width_ * height_);
  return true;
}

bool frolova_e_sobel_filter_stl::SobelFilterSTL::ValidationImpl() {
  // Check equality of counts elements
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

  return true;
}

bool frolova_e_sobel_filter_stl::SobelFilterSTL::RunImpl() {
  std::vector<int> grayscale_image(width_ * height_);
  int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;

  auto grayscale_task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      grayscale_image[i] =
          static_cast<int>((0.299 * picture_[i].R) + (0.587 * picture_[i].G) + (0.114 * picture_[i].B));
    }
  };

  size_t total_pixels = width_ * height_;
  size_t actual_threads_grayscale = std::min(static_cast<size_t>(num_threads), total_pixels);
  size_t chunk_size = (total_pixels + actual_threads_grayscale - 1) / actual_threads_grayscale;

  for (size_t i = 0; i < actual_threads_grayscale; ++i) {
    size_t start = i * chunk_size;
    size_t end = std::min(start + chunk_size, total_pixels);
    if (start < end) {
      threads.emplace_back(grayscale_task, start, end);
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  auto sobel_task = [&](size_t y_start, size_t y_end) {
    for (size_t y = y_start; y < y_end; ++y) {
      for (size_t x = 0; x < width_; ++x) {
        size_t base_idx = (y * width_) + x;

        int p00 = GetPixelSafe(grayscale_image, x - 1, y - 1, width_, height_);
        int p01 = GetPixelSafe(grayscale_image, x, y - 1, width_, height_);
        int p02 = GetPixelSafe(grayscale_image, x + 1, y - 1, width_, height_);
        int p10 = GetPixelSafe(grayscale_image, x - 1, y, width_, height_);
        int p11 = GetPixelSafe(grayscale_image, x, y, width_, height_);
        int p12 = GetPixelSafe(grayscale_image, x + 1, y, width_, height_);
        int p20 = GetPixelSafe(grayscale_image, x - 1, y + 1, width_, height_);
        int p21 = GetPixelSafe(grayscale_image, x, y + 1, width_, height_);
        int p22 = GetPixelSafe(grayscale_image, x + 1, y + 1, width_, height_);

        int res_x = (-1 * p00) + (0 * p01) + (1 * p02) + (-2 * p10) + (0 * p11) + (2 * p12) + (-1 * p20) + (0 * p21) +
                    (1 * p22);
        int res_y = (-1 * p00) + (-2 * p01) + (-1 * p02) + (0 * p10) + (0 * p11) + (0 * p12) + (1 * p20) + (2 * p21) +
                    (1 * p22);

        int gradient = static_cast<int>(sqrt((res_x * res_x) + (res_y * res_y)));
        res_image_[base_idx] = std::clamp(gradient, 0, 255);
      }
    }
  };

  size_t actual_threads_sobel = std::min(static_cast<size_t>(num_threads), height_);
  size_t rows_per_thread = (height_ + actual_threads_sobel - 1) / actual_threads_sobel;

  for (size_t i = 0; i < actual_threads_sobel; ++i) {
    size_t y_start = i * rows_per_thread;
    size_t y_end = std::min(y_start + rows_per_thread, height_);
    if (y_start < y_end) {
      threads.emplace_back(sobel_task, y_start, y_end);
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return true;
}

bool frolova_e_sobel_filter_stl::SobelFilterSTL::PostProcessingImpl() {
  std::ranges::copy(res_image_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
