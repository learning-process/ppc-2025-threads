#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/kolokolova_d_integral_simpson_method_tbb/include/ops_tbb.hpp"

TEST(kolokolova_d_integral_simpson_method_tbb, test_pipeline_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {130, 130, 130};
  std::vector<int> bord = {1, 11, 2, 10, 0, 10};
  double func_result = 0.0;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_tbb->inputs_count.emplace_back(step.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_tbb->inputs_count.emplace_back(bord.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_tbb->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_tbb = std::make_shared<kolokolova_d_integral_simpson_method_tbb::TestTaskTBB>(task_data_tbb, func);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(1, 1);
}

TEST(kolokolova_d_integral_simpson_method_tbb, test_task_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {130, 130, 130};
  std::vector<int> bord = {1, 11, 2, 10, 0, 10};
  double func_result = 0.0;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_tbb->inputs_count.emplace_back(step.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_tbb->inputs_count.emplace_back(bord.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_tbb->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_tbb = std::make_shared<kolokolova_d_integral_simpson_method_tbb::TestTaskTBB>(task_data_tbb, func);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(1, 1);
}
