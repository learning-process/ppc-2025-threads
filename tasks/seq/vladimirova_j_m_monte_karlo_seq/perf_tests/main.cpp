
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/vladimirova_j_m_monte_karlo_seq/include/ops_seq.hpp"

namespace {
bool PiVal314(std::vector<double> arr, size_t size = 2) {
  double x = arr[0];
  double y = arr[1];
  return (((x * x) + (y * y) - 1) <= 0);
};
} 

TEST(vladimirova_j_m_monte_karlo_seq, test_pipeline_run) {
  // Create data
  std::vector<double> val_b = {-1, 1, -1, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(val_b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(PiVal314));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(200000));
  task_data_seq->inputs_count.emplace_back(val_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<vladimirova_j_m_monte_karlo_seq::TestTaskSequential>(task_data_seq);
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE((3.14 - out[0]) < 0.005);
}

TEST(vladimirova_j_m_monte_karlo_seq, test_task_run) {
  // Create data
  std::vector<double> val_b = {-1, 1, -1, 1};
  std::vector<double> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(val_b.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(PiVal314));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(200000));
  task_data_seq->inputs_count.emplace_back(val_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<vladimirova_j_m_monte_karlo_seq::TestTaskSequential>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_TRUE((3.14 - out[0]) < 0.005);
  ;
}
