#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "random"

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/frolova_e_Sobel_filter/include/ops_omp.hpp"

namespace {
std::vector<int> GenRgbPicture(size_t width, size_t height, size_t seed) {
  std::vector<int> image(width * height * 3);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> rgb(0, 255);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      size_t index = (y * width + x) * 3;
      image[index] = rgb(gen);
      image[index + 1] = rgb(gen);
      image[index + 2] = rgb(gen);
    }
  }

  return image;
}

std::vector<frolova_e_sobel_filter_omp::RGB> ConvertToRGB(const std::vector<int> &pict) {
  std::vector<frolova_e_sobel_filter_omp::RGB> picture;
  size_t pixel_count = pict.size() / 3;

  for (size_t i = 0; i < pixel_count; i++) {
    frolova_e_sobel_filter_omp::RGB pixel;
    pixel.R = pict[i * 3];
    pixel.G = pict[(i * 3) + 1];
    pixel.B = pict[(i * 3) + 2];

    picture.push_back(pixel);
  }
  return picture;
}

}  // namespace

TEST(frolova_e_sobel_filter_omp, test_pipeline_run) {
  std::vector<int> value = {2000, 2000};
  std::vector<int> pict = GenRgbPicture(2000, 2000, 0);

  std::vector<int> res(4000000, 0);

  std::vector<int> reference(4000000, 0);

  // Create task_data
  auto task_data_ = std::make_shared<ppc::core::TaskData>();

  task_data_->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data_->inputs_count.emplace_back(value.size());

  task_data_->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_->inputs_count.emplace_back(pict.size());

  task_data_->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_->outputs_count.emplace_back(res.size());

  // Create Task
  auto test_task_omp = std::make_shared<frolova_e_sobel_filter_omp::SobelFilterOmp>(task_data_);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<frolova_e_sobel_filter_omp::RGB> picture = ConvertToRGB(pict);
  std::vector<int> gray_scale_image =
      frolova_e_sobel_filter_omp::ToGrayScaleImg(picture, static_cast<size_t>(value[0]), static_cast<size_t>(value[1]));

  const std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  for (size_t y = 0; y < static_cast<size_t>(value[0]); y++) {
    for (size_t x = 0; x < static_cast<size_t>(value[1]); x++) {
      int res_x = 0;
      int res_y = 0;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int px = static_cast<int>(x) + kx;
          int py = static_cast<int>(y) + ky;

          int pixel_value = 0;

          if (px >= 0 && px < static_cast<int>(value[0]) && py >= 0 && py < static_cast<int>(value[1])) {
            pixel_value = gray_scale_image[(py * value[0]) + px];
          }

          size_t kernel_ind = ((ky + 1) * 3) + (kx + 1);
          res_x += pixel_value * gx[kernel_ind];
          res_y += pixel_value * gy[kernel_ind];
        }
      }
      int gradient = static_cast<int>(sqrt((res_x * res_x) + (res_y * res_y)));
      reference[(y * value[0]) + x] = std::clamp(gradient, 0, 255);
    }
  }

  ASSERT_EQ(reference, res);
}

TEST(frolova_e_sobel_filter_omp, test_task_run) {
  std::vector<int> value = {2000, 2000};
  std::vector<int> pict = GenRgbPicture(2000, 2000, 0);

  std::vector<int> res(4000000, 0);

  std::vector<int> reference(4000000, 0);

  // Create task_data
  auto task_data_ = std::make_shared<ppc::core::TaskData>();

  task_data_->inputs.emplace_back(reinterpret_cast<uint8_t *>(value.data()));
  task_data_->inputs_count.emplace_back(value.size());

  task_data_->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_->inputs_count.emplace_back(pict.size());

  task_data_->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_->outputs_count.emplace_back(res.size());

  // Create Task
  auto test_task_omp = std::make_shared<frolova_e_sobel_filter_omp::SobelFilterOmp>(task_data_);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<frolova_e_sobel_filter_omp::RGB> picture = ConvertToRGB(pict);
  std::vector<int> gray_scale_image =
      frolova_e_sobel_filter_omp::ToGrayScaleImg(picture, static_cast<size_t>(value[0]), static_cast<size_t>(value[1]));

  const std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  for (size_t y = 0; y < static_cast<size_t>(value[0]); y++) {
    for (size_t x = 0; x < static_cast<size_t>(value[1]); x++) {
      int res_x = 0;
      int res_y = 0;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int px = static_cast<int>(x) + kx;
          int py = static_cast<int>(y) + ky;

          int pixel_value = 0;

          if (px >= 0 && px < static_cast<int>(value[0]) && py >= 0 && py < static_cast<int>(value[1])) {
            pixel_value = gray_scale_image[(py * value[0]) + px];
          }

          size_t kernel_ind = ((ky + 1) * 3) + (kx + 1);
          res_x += pixel_value * gx[kernel_ind];
          res_y += pixel_value * gy[kernel_ind];
        }
      }
      int gradient = static_cast<int>(sqrt((res_x * res_x) + (res_y * res_y)));
      reference[(y * value[0]) + x] = std::clamp(gradient, 0, 255);
    }
  }

  ASSERT_EQ(reference, res);
}
