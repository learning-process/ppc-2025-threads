#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/kondratev_ya_ccs_complex_multiplication/include/ops_omp.hpp"

TEST(kondratev_ya_ccs_complex_multiplication_omp, test_pipeline_run) {
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix c({kCount, kCount});

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_omp->inputs_count.emplace_back(2);

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
  task_data_omp->outputs_count.emplace_back(1);

  auto test_task_ompuential =
      std::make_shared<kondratev_ya_ccs_complex_multiplication_omp::TestTaskOMPuential>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_ompuential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(kondratev_ya_ccs_complex_multiplication_omp, test_task_run) {
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix c({kCount, kCount});

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_omp->inputs_count.emplace_back(2);

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
  task_data_omp->outputs_count.emplace_back(1);

  auto test_task_ompuential =
      std::make_shared<kondratev_ya_ccs_complex_multiplication_omp::TestTaskOMPuential>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_ompuential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
