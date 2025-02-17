#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/lopatin_i_monte_carlo/include/lopatinMonteCarloSeq.hpp"

namespace lopatin_i_monte_carlo_seq {

std::vector<double> generateBounds(int dimensions, double min_val = 0.0, double max_val = 1.0) {
  std::vector<double> bounds;
  for (int i = 0; i < dimensions; ++i) {
    bounds.push_back(min_val);
    bounds.push_back(max_val);
  }
  return bounds;
}
}  // namespace lopatin_i_monte_carlo_seq

TEST(lopatin_i_monte_carlo_seq, test_pipeline_run) {
  const int dimensions = 5;
  const int iterations = 10000000;

  std::vector<double> bounds = lopatin_i_monte_carlo_seq::generateBounds(dimensions, -3.0, 3.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // exp(x1 + x2 + x3 + x4 + x5)
  auto function = [](const std::vector<double>& x) { return std::exp(x[0] + x[1] + x[2] + x[3] + x[4]); };

  auto test_task = std::make_shared<lopatin_i_monte_carlo_seq::TestTaskSequential>(task_data, function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(lopatin_i_monte_carlo_seq, test_task_run) {
  const int dimensions = 5;
  const int iterations = 10000000;

  std::vector<double> bounds = lopatin_i_monte_carlo_seq::generateBounds(dimensions, -3.0, 3.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // exp(x1 + x2 + x3 + x4 + x5)
  auto function = [](const std::vector<double>& x) { return std::exp(x[0] + x[1] + x[2] + x[3] + x[4]); };

  auto test_task = std::make_shared<lopatin_i_monte_carlo_seq::TestTaskSequential>(task_data, function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}