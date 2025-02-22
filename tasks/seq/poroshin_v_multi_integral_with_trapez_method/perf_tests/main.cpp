#include <gtest/gtest.h>

#include <numbers>

#include "core/perf/include/perf.hpp"
#include "seq/poroshin_v_multi_integral_with_trapez_method/include/ops_seq.hpp"

namespace poroshin_v_multi_integral_with_trapez_method_seq {
double f3advanced(std::vector<double> &arguments) {
  return sin(arguments.at(0)) * tan(arguments.at(1)) * log(arguments.at(2));
}
}  // namespace poroshin_v_multi_integral_with_trapez_method_seq

TEST(poroshin_v_multi_integral_with_trapez_method_seq, test_pipeline_run) {
  std::vector<int> n = {300, 300, 300};
  std::vector<double> a = {0.8, 1.9, 2.9};
  std::vector<double> b = {1.0, 2.0, 3.0};
  std::vector<double> out(1);
  double eps = 1e-6;
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential>(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::f3advanced);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, test_task_run) {
  std::vector<int> n = {300, 300, 300};
  std::vector<double> a = {0.8, 1.9, 2.9};
  std::vector<double> b = {1.0, 2.0, 3.0};
  std::vector<double> out(1);
  double eps = 1e-6;
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential>(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::f3advanced);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}