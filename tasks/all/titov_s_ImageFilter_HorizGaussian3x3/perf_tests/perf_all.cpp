#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/titov_s_ImageFilter_HorizGaussian3x3/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
constexpr size_t kWidth = 15000;
constexpr size_t kHeight = 15000;

void InitializeInputData(std::vector<double>& input) {
  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      input[(i * kWidth) + j] = (j % 3 == 0) ? 100.0 : 0.0;
    }
  }
}

void InitializeExpectedData(std::vector<double>& expected) {
  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      expected[(i * kWidth) + j] = (j == kWidth - 1) ? 0.0 : (j % 3 == 0) ? 50.0 : 25.0;
    }
  }
}

void VerifyResults(const std::vector<double>& output, const std::vector<double>& expected,
                   const boost::mpi::communicator& world) {
  if (world.rank() != 0) return;

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      ASSERT_NEAR(output[(i * kWidth) + j], expected[(i * kWidth) + j], 1e-6);
    }
  }
}

std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<double>& input, const std::vector<int>& kernel,
                                                    std::vector<double>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size() * sizeof(double));

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(kernel.data())));
  task_data->inputs_count.emplace_back(kernel.size() * sizeof(int));

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size() * sizeof(double));

  return task_data;
}

std::shared_ptr<ppc::core::PerfAttr> CreatePerfAttr() {
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  return perf_attr;
}

void RunPerformanceTest(std::shared_ptr<ppc::core::Task> test_task, std::shared_ptr<ppc::core::PerfAttr> perf_attr,
                        std::shared_ptr<ppc::core::PerfResults> perf_results, bool pipeline) {
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  if (pipeline) {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
  } else {
    perf_analyzer->TaskRun(perf_attr, perf_results);
  }
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
}  // namespace

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> input(kWidth * kHeight);
  std::vector<double> output(kWidth * kHeight);
  std::vector<double> expected(kWidth * kHeight);
  std::vector<int> kernel = {1, 2, 1};

  InitializeInputData(input);
  InitializeExpectedData(expected);

  auto task_data = CreateTaskData(input, kernel, output);
  auto test_task = std::make_shared<titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL>(task_data);
  auto perf_attr = CreatePerfAttr();
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  RunPerformanceTest(test_task, perf_attr, perf_results, true);
  VerifyResults(output, expected, world);
}

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> input(kWidth * kHeight);
  std::vector<double> output(kWidth * kHeight);
  std::vector<double> expected(kWidth * kHeight);
  std::vector<int> kernel = {1, 2, 1};

  InitializeInputData(input);
  InitializeExpectedData(expected);

  auto task_data = CreateTaskData(input, kernel, output);
  auto test_task = std::make_shared<titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL>(task_data);
  auto perf_attr = CreatePerfAttr();
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  RunPerformanceTest(test_task, perf_attr, perf_results, false);
  VerifyResults(output, expected, world);
}