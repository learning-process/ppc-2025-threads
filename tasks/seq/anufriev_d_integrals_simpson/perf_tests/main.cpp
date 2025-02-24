#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/anufriev_d_integrals_simpson/include/ops_seq.hpp"

TEST(anufriev_d_integrals_simpson_seq, test_pipeline_run) {
  double ax = 0.0;
  double bx = 1.0;
  double ay = 0.0;
  double by = 1.0;
  int nx = 10000;
  int ny = 10000;
  int func_code = 0;

  std::vector<double> in = {ax, bx, (double)nx, ay, by, (double)ny, (double)func_code};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto task_seq = std::make_shared<anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 15;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  double result = out[0];
  ASSERT_NEAR(result, 2.0 / 3.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_seq_perf, test_task_run) {
  double ax = 0.0;
  double bx = 1.0;
  double ay = 0.0;
  double by = 1.0;
  int nx = 10000;
  int ny = 10000;
  int func_code = 0;

  std::vector<double> in = {ax, bx, (double)nx, ay, by, (double)ny, (double)func_code};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto task_seq = std::make_shared<anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 15;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_seq);

  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  double result = out[0];
  ASSERT_NEAR(result, 2.0 / 3.0, 1e-3);
}