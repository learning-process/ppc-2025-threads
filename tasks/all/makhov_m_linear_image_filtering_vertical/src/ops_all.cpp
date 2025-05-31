#include "all/makhov_m_linear_image_filtering_vertical/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

using std::uint32_t;
using std::uint8_t;
using std::vector;

void makhov_m_linear_image_filtering_vertical_all::BlurColumn(const uint8_t* src, uint8_t* dst, int width, int height,
                                                              int x) {
  const std::vector<float> kernel = {0.25F, 0.5F, 0.25F};
  const int kernel_radius = 1;

  if (x < 0 || x >= width) {
    return;  // Некорректный индекс
  }

#pragma omp parallel for
  for (int y = 0; y < height; ++y) {
    float sum_r = 0.0F;
    float sum_g = 0.0F;
    float sum_b = 0.0F;

    for (int k = -kernel_radius; k <= kernel_radius; ++k) {
      int ny = std::clamp(y + k, 0, height - 1);
      size_t pixel_idx = (ny * width + x) * 3;
      if (pixel_idx + 2 >= (static_cast<size_t>(width) * static_cast<size_t>(height) * 3)) {
        continue;  // Защита от выхода за границы
      }

      sum_r += static_cast<float>(src[pixel_idx]) * kernel[k + kernel_radius];
      sum_g += static_cast<float>(src[pixel_idx + 1]) * kernel[k + kernel_radius];
      sum_b += static_cast<float>(src[pixel_idx + 2]) * kernel[k + kernel_radius];
    }

    size_t out_idx = (y * width + x) * 3;
    if (out_idx + 2 < (static_cast<size_t>(width) * static_cast<size_t>(height) * 3)) {
      dst[out_idx] = static_cast<uint8_t>(std::clamp(sum_r, 0.0F, 255.0F));
      dst[out_idx + 1] = static_cast<uint8_t>(std::clamp(sum_g, 0.0F, 255.0F));
      dst[out_idx + 2] = static_cast<uint8_t>(std::clamp(sum_b, 0.0F, 255.0F));
    }
  }
}

bool makhov_m_linear_image_filtering_vertical_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count.size() >= 2 && !task_data->outputs_count.empty() && !task_data->inputs.empty() &&
           !task_data->outputs.empty() && (task_data->inputs_count[0] >= 3 || task_data->inputs_count[1] >= 3) &&
           (task_data->inputs_count[0] * task_data->inputs_count[1] * 3 == task_data->outputs_count[0]);
  }
  return true;
}

bool makhov_m_linear_image_filtering_vertical_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    width_ = static_cast<std::uint32_t>(task_data->inputs_count[0]);
    height_ = static_cast<std::uint32_t>(task_data->inputs_count[1]);
    input_size_ = width_ * height_ * 3;

    if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
      throw std::runtime_error("Input data is missing");
    }

    // Проверка на переполнение
    const size_t max_pixels = 1000000;  // Пример: ограничение до 1 млн пикселей
    if (static_cast<size_t>(width_) * height_ > max_pixels) {
      throw std::runtime_error("Image size exceeds limit");
    }

    input_image_.assign(task_data->inputs[0], task_data->inputs[0] + input_size_);

    output_image_.resize(input_size_);  // Важно: выделение памяти заранее
  }
  return true;
}

bool makhov_m_linear_image_filtering_vertical_all::TestTaskALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  // 1) Рассылаем width, height и весь input_image на все ранги
  boost::mpi::broadcast(world_, width_, 0);
  boost::mpi::broadcast(world_, height_, 0);
  if (rank != 0) {
    input_image_.resize(width_ * height_ * 3);
  }
  boost::mpi::broadcast(world_, input_image_, 0);

  // 2) Разбиваем столбцы поровну
  int width_i = static_cast<int>(width_);
  int size_i = size;

  std::vector<int> counts(size_i, width_i / size_i);
  for (int i = 0, rem = width_i % size_i; i < rem; ++i) {
    counts[i]++;
  }
  std::vector<int> displs(size, 0);
  std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);

  int local_cols = counts[rank];
  int offset = displs[rank];

  // 3) Локальный буфер и свой блуp-код
  std::vector<uint8_t> local_out(local_cols * height_ * 3, 0U);
  const float kernel[3] = {0.25F, 0.5F, 0.25F};

#pragma omp parallel for
  for (int cx = 0; cx < local_cols; ++cx) {
    for (int y = 0; y < static_cast<int>(height_); ++y) {
      float sum_r = 0.0F;
      float sum_g = 0.0F;
      float sum_b = 0.0F;
      // вертикальное размытие
      for (int k = -1; k <= 1; ++k) {
        int yy = std::clamp(y + k, 0, static_cast<int>(height_) - 1);
        size_t in_idx = (static_cast<size_t>(yy) * width_ + (cx + offset)) * 3;
        sum_r += static_cast<float>(input_image_[in_idx + 0]) * kernel[k + 1];
        sum_g += static_cast<float>(input_image_[in_idx + 1]) * kernel[k + 1];
        sum_b += static_cast<float>(input_image_[in_idx + 2]) * kernel[k + 1];
      }
      size_t out_idx = (static_cast<size_t>(y) * local_cols + cx) * 3;
      auto clamp8 = [](float v) { return static_cast<uint8_t>(std::clamp(v, 0.0F, 255.0F)); };
      local_out[out_idx + 0] = clamp8(sum_r);
      local_out[out_idx + 1] = clamp8(sum_g);
      local_out[out_idx + 2] = clamp8(sum_b);
    }
  }

  // 4) Собираем все local_out в вектор векторов на root-е
  std::vector<std::vector<uint8_t>> gathered;
  boost::mpi::gather(world_, local_out, gathered, 0);

  // 5) На root-е склеиваем их в output_image
  if (rank == 0) {
    output_image_.resize(width_ * height_ * 3);
    std::vector<int> counts_bytes(size);
    std::vector<int> displs_bytes(size);
    for (int i = 0; i < size; ++i) {
      counts_bytes[i] = counts[i] * static_cast<int>(height_) * 3;
      displs_bytes[i] = displs[i] * static_cast<int>(height_) * 3;
    }
    for (int i = 0; i < size; ++i) {
      std::memcpy(output_image_.data() + displs_bytes[i], gathered[i].data(), counts_bytes[i]);
    }
  }

  return true;
}

bool makhov_m_linear_image_filtering_vertical_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::memcpy(task_data->outputs[0], output_image_.data(), output_image_.size());
  }
  return true;
}
