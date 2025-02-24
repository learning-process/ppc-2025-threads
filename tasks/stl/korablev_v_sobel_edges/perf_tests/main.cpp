#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/korablev_v_sobel_edges/include/ops_stl.hpp"

TEST(korablev_v_sobel_edges_stl, test_pipeline_run) {
  const std::size_t height = 15'000;
  const std::size_t width = 1'000;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 255);

  std::vector<uint8_t> in(width * height * 3);
  std::ranges::generate(in.begin(), in.end(), [&] { return dist(gen); });
  std::vector<uint8_t> out(width * height * 3);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  task_data->outputs = {out.data()};
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  auto task = std::make_shared<korablev_v_sobel_edges_stl::TestTask>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(korablev_v_sobel_edges_stl, test_task_run) {
  const std::size_t width = 5'000;
  const std::size_t height = width;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 255);

  std::vector<uint8_t> in(width * height * 3);
  std::ranges::generate(in.begin(), in.end(), [&] { return dist(gen); });
  std::vector<uint8_t> out(width * height * 3);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  task_data->outputs = {out.data()};
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  auto task = std::make_shared<korablev_v_sobel_edges_stl::TestTask>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}