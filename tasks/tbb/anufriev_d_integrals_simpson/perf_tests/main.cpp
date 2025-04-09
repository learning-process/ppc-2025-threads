#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

#include "tbb/anufriev_d_integrals_simpson/include/ops_tbb.hpp"

TEST(anufriev_d_integrals_simpson_tbb, test_pipeline_run) {
  std::vector<double> in = {2, 0.0, 1.0, 2000, 0.0, 1.0, 2000, 0};
  std::vector<double> out(1, 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_tbb->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  task_data_tbb->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.push_back(static_cast<std::uint32_t>(out.size() * sizeof(double)));

  auto task_tbb = std::make_shared<anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_tbb);

  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  double result = out[0];
  EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_tbb, test_task_run) {
  std::vector<double> in = {2, 0.0, 1.0, 2000, 0.0, 1.0, 2000, 0};
  std::vector<double> out(1, 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_tbb->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  task_data_tbb->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_tbb->outputs_count.push_back(static_cast<std::uint32_t>(out.size() * sizeof(double)));

  auto task_tbb = std::make_shared<anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_tbb);

  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  double result = out[0];
  EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
}