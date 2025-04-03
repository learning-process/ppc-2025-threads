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

namespace {
MatrixStructure GenerateRandomMatrix(uint32_t num_rows, uint32_t num_cols, double non_zero_percentage) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> distribution(-10000, 10000);
  MatrixStructure result{
      .num_rows = num_rows, .num_cols = num_cols, .task_data = std::vector<std::complex<double>>(num_rows * num_cols)};
  std::ranges::generate(result.task_data, [&]() {
    const auto value = distribution(rng);
    const auto real_component =
        (value < (distribution.min() + ((distribution.max() - distribution.min()) * non_zero_percentage))) ? value : 0;

    std::complex<double> complex_number;
    complex_number.real(real_component);
    if (real_component != 0.0) {
      complex_number.imag(distribution(rng));
    }

    return complex_number;
  });
  return result;
}
}  // namespace

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_pipeline_run) {
  auto matrix_left = GenerateRandomMatrix(730, 730, 0.22);
  auto matrix_right = GenerateRandomMatrix(730, 730, 0.22);

  SparseMatrixFormat compressed_row_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat compressed_row_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat compressed_row_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&compressed_row_left),
                       reinterpret_cast<uint8_t *>(&compressed_row_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&compressed_row_result)};
  task_data->outputs_count = {1};

  auto task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(task_data);

  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  auto performance_analyzer = std::make_shared<ppc::core::Perf>(task);
  performance_analyzer->PipelineRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  EXPECT_EQ(ConvertFromCRS(compressed_row_result), MatrixMultiply(matrix_left, matrix_right));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_task_run) {
  auto matrix_left = GenerateRandomMatrix(730, 730, 0.22);
  auto matrix_right = GenerateRandomMatrix(730, 730, 0.22);

  SparseMatrixFormat compressed_row_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat compressed_row_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat compressed_row_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&compressed_row_left),
                       reinterpret_cast<uint8_t *>(&compressed_row_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&compressed_row_result)};
  task_data->outputs_count = {1};

  auto task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(task_data);

  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  auto performance_analyzer = std::make_shared<ppc::core::Perf>(task);
  performance_analyzer->TaskRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  EXPECT_EQ(ConvertFromCRS(compressed_row_result), MatrixMultiply(matrix_left, matrix_right));
}