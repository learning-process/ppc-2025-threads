#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

TEST(sequential_zolotareva_a_sle_gradient_method_seq, test_pipeline_run) {
  const int n = 1000;
  std::vector<double> a(n * n, 0);
  std::vector<double> b(n, 0);
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(x.size());

  auto test_task_sequential = std::make_shared<zolotareva_a_sle_gradient_method_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(b, x);
}

TEST(sequential_zolotareva_a_sle_gradient_method_seq, test_task_run) {
  const int n = 1000;
  std::vector<double> a(n * n, 0);
  std::vector<double> b(n, 0);
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(x.size());

  auto test_task_sequential = std::make_shared<zolotareva_a_sle_gradient_method_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(b, x);
}
