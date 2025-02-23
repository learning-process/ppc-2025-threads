#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/chernykh_a_multidimensional_integral_rectangle/include/ops_seq.hpp"

namespace {

using namespace chernykh_a_multidimensional_integral_rectangle_seq;

enum class RunType : uint8_t { kTask, kPipeline };

void RunTask(const RunType run_type, const Func& func, BoundsPerDim& bounds_per_dim, StepsPerDim& steps_per_dim,
             const double want) {
  double output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds_per_dim.data()));
  task_data->inputs_count.emplace_back(bounds_per_dim.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps_per_dim.data()));
  task_data->inputs_count.emplace_back(steps_per_dim.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<SequentialTask>(task_data, func);

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&]() -> double {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  switch (run_type) {
    case RunType::kPipeline:
      perf_analyzer->PipelineRun(perf_attributes, perf_results);
      break;
    case RunType::kTask:
      perf_analyzer->TaskRun(perf_attributes, perf_results);
      break;
  }

  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_NEAR(want, output, 1e-4);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, test_pipeline_run) {
  auto func = [](const Point& point) -> double {
    return std::exp(-point[0] - point[1] - point[2]) * std::sin(point[0]) * std::sin(point[1]) * std::sin(point[2]);
  };
  auto bounds_per_dim = BoundsPerDim{
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi},
  };
  auto steps_per_dim = StepsPerDim{125, 125, 125};
  double want = std::pow((1.0 + std::exp(-std::numbers::pi)) / 2.0, 3);
  RunTask(RunType::kPipeline, func, bounds_per_dim, steps_per_dim, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, test_task_run) {
  auto func = [](const Point& point) -> double {
    return std::exp(-point[0] - point[1] - point[2]) * std::sin(point[0]) * std::sin(point[1]) * std::sin(point[2]);
  };
  auto bounds_per_dim = BoundsPerDim{
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi},
  };
  auto steps_per_dim = StepsPerDim{125, 125, 125};
  double want = std::pow((1.0 + std::exp(-std::numbers::pi)) / 2.0, 3);
  RunTask(RunType::kTask, func, bounds_per_dim, steps_per_dim, want);
}

}  // namespace
