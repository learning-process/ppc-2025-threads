#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

namespace yasakova_t_sparse_matrix_multiplication_omp {
MatrixStructure RandMatrix(uint32_t num_rows, uint32_t num_cols, double non_zero_percentage) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> distr(-10000, 10000);
  MatrixStructure result{
      .num_rows = num_rows, .num_cols = num_cols, .task_data = std::vector<std::complex<double>>(num_rows * num_cols)};
  std::ranges::generate(result.task_data, [&]() {
    const auto value = distr(gen);
    const auto real_part = (value < (distr.min() + ((distr.max() - distr.min()) * non_zero_percentage))) ? value : 0;

    std::complex<double> complex_num;
    complex_num.real(real_part);
    if (real_part != 0.0) {
      complex_num.imag(distr(gen));
    }

    return complex_num;
  });
  return result;
}
}  // namespace yasakova_t_sparse_matrix_multiplication_omp

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_pipeline_run) {
  auto matrix_left = yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(730, 730, 0.22);
  auto matrix_right = yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(730, 730, 0.22);

  SparseMatrixFormat crs_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat crs_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat crs_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&crs_left), reinterpret_cast<uint8_t *>(&crs_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&crs_result)};
  task_data->outputs_count = {1};

  auto task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_EQ(ConvertFromCRS(crs_result), MatrixMultiply(matrix_left, matrix_right));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_task_run) {
  auto matrix_left = yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(730, 730, 0.22);
  auto matrix_right = yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(730, 730, 0.22);

  SparseMatrixFormat crs_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat crs_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat crs_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&crs_left), reinterpret_cast<uint8_t *>(&crs_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&crs_result)};
  task_data->outputs_count = {1};

  auto task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_EQ(ConvertFromCRS(crs_result), MatrixMultiply(matrix_left, matrix_right));
}