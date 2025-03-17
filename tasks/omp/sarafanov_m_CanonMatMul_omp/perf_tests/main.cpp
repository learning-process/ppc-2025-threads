#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/sarafanov_m_CanonMatMul_omp/include/ops_omp.hpp"

TEST(sarafanov_m_canon_mat_mul_omp, test_pipeline_run) {
  constexpr size_t kCount = 122500;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = sarafanov_m_canon_mat_mul_omp::GenerateRandomData(static_cast<int>(kCount));
  auto single_matrix = sarafanov_m_canon_mat_mul_omp::GenerateSingleMatrix(static_cast<int>(kCount));
  std::vector<double> out(kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  task_data_omp->inputs_count.emplace_back(kCount);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(kCount);
  auto test_task_omp = std::make_shared<sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP>(task_data_omp);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_task_run) {
  constexpr size_t kCount = 122500;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = sarafanov_m_canon_mat_mul_omp::GenerateRandomData(static_cast<int>(kCount));
  auto single_matrix = sarafanov_m_canon_mat_mul_omp::GenerateSingleMatrix(static_cast<int>(kCount));
  std::vector<double> out(kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  task_data_omp->inputs_count.emplace_back(kCount);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(kCount);

  auto test_task_omp = std::make_shared<sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}
