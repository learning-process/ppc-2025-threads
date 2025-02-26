#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/shurigin_s_integrals_square_OMP/include/ops_omp.hpp"

namespace shurigin_s_integrals_square_omp {

TEST(shurigin_s_integrals_square_omp, test_pipeline_run) {
  double down_limit = -1.0;
  double up_limit = 1.0;
  int count = 1000000;
  std::vector<double> inputs{down_limit, up_limit, static_cast<double>(count)};
  double result = 0.0;

  auto f = [](double x) { return std::cos(x * x) * (1 + x * x); };

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  task_data_seq->inputs_count.emplace_back(inputs.size() * sizeof(double));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_seq->outputs_count.emplace_back(sizeof(double));

  auto test_task_sequential = std::make_shared<Integral>(task_data_seq);
  test_task_sequential->SetFunction(f);

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
}

TEST(shurigin_s_integrals_square_omp, test_task_run) {
  double down_limit = -1.0;
  double up_limit = 1.0;
  int count = 1000000;
  std::vector<double> inputs{down_limit, up_limit, static_cast<double>(count)};
  double result = 0.0;

  auto f = [](double x) { return std::cos(x * x) * (1 + x * x); };

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
  task_data_seq->inputs_count.emplace_back(inputs.size() * sizeof(double));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_seq->outputs_count.emplace_back(sizeof(double));

  auto test_task_sequential = std::make_shared<Integral>(task_data_seq);
  test_task_sequential->SetFunction(f);

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
}

}  // namespace shurigin_s_integrals_square_omp
