#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/makhov_m_linear_image_filtering_vertical/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<uint8_t> GenerateRandomImage(int height, int width) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  int size = height * width * 3;
  std::vector<uint8_t> image(size);

  for (int i = 0; i < size; ++i) {
    image[i] = dis(gen);
  }

  return image;
}

void ApplyNaiveGaussianBlur(std::vector<uint8_t>& image, int width, int height) {
  std::vector<uint8_t> original = image;
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int sum_r = 0;
      int sum_g = 0;
      int sum_b = 0;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          std::size_t idx = ((y + ky) * width + (x + kx)) * 3;
          sum_r += original[idx];
          sum_g += original[idx + 1];
          sum_b += original[idx + 2];
        }
      }

      std::size_t idx = (y * width + x) * 3;
      image[idx] = static_cast<uint8_t>(sum_r / 9);
      image[idx + 1] = static_cast<uint8_t>(sum_g / 9);
      image[idx + 2] = static_cast<uint8_t>(sum_b / 9);
    }
  }
}

void ValidateOutputImage(const std::vector<uint8_t>& image) {
  for (uint8_t pixel : image) {
    EXPECT_GE(pixel, 0);
    EXPECT_LE(pixel, 255);
  }
}

void PrepareTaskData(std::shared_ptr<ppc::core::TaskData>& data, std::vector<uint8_t>& input,
                     std::vector<uint8_t>& output, int width, int height) {
  data->inputs.emplace_back(input.data());
  data->inputs_count = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
  data->outputs.emplace_back(output.data());
  data->outputs_count.emplace_back(static_cast<std::uint32_t>(output.size()));
}
}  // namespace

TEST(makhov_m_linear_image_filtering_vertical_all, test_pipeline_run) {
  boost::mpi::communicator world;

  const int width = 1000;
  const int height = 1000;

  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);
  std::vector<uint8_t> reference_image(input_image);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    ApplyNaiveGaussianBlur(reference_image, width, height);
    PrepareTaskData(task_data_mpi, input_image, output_image, width, height);
  }

  auto test_task_mpi = std::make_shared<makhov_m_linear_image_filtering_vertical_all::TestTaskALL>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    ValidateOutputImage(output_image);
  }
}

TEST(makhov_m_linear_image_filtering_vertical_all, test_task_run) {
  boost::mpi::communicator world;

  const int width = 1000;
  const int height = 1000;

  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);
  std::vector<uint8_t> reference_image(input_image);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    ApplyNaiveGaussianBlur(reference_image, width, height);
    PrepareTaskData(task_data_mpi, input_image, output_image, width, height);
  }

  auto test_task_mpi = std::make_shared<makhov_m_linear_image_filtering_vertical_all::TestTaskALL>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    ValidateOutputImage(output_image);
  }
}
