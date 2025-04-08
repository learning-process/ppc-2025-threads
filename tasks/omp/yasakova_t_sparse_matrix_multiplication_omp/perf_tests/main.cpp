#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

namespace {
SparseMatrixFormat CreateRandomSparseMatrix(uint32_t size, uint32_t non_zero_elements) {
  SparseMatrixFormat matrix;
  matrix.columns = size;
  matrix.row_pointers.resize(size + 1);
  
  std::srand(std::time(nullptr));
  for (uint32_t i = 0; i < non_zero_elements; i++) {
    uint32_t row = rand() % size;
    uint32_t col = rand() % size;
    std::complex<double> value(-50 + (rand() % 100), -50 + (rand() % 100));
    
    // Simple way to generate sparse matrix (not optimal but works for tests)
    matrix.task_data.push_back(value);
    matrix.column_indices.push_back(col);
    matrix.row_pointers[row + 1]++;
  }

  // Calculate row pointers
  for (uint32_t i = 1; i <= size; i++) {
    matrix.row_pointers[i] += matrix.row_pointers[i - 1];
  }

  return matrix;
}
}  // namespace

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_pipeline_run) {
  const uint32_t matrix_size = 400;
  const uint32_t non_zero_elements = 5000;
  const uint32_t num_runs = 10;

  // Create random sparse matrices
  auto sparse_matrix_a = CreateRandomSparseMatrix(matrix_size, non_zero_elements);
  auto sparse_matrix_b = CreateRandomSparseMatrix(matrix_size, non_zero_elements);

  // Create task data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sparse_matrix_a));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sparse_matrix_b));
  task_data->inputs_count.emplace_back(1);
  
  SparseMatrixFormat result_matrix;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_matrix));
  task_data->outputs_count.emplace_back(1);

  // Create Task
  auto omp_task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(task_data);

  // Create Performance attributes
  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = num_runs;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(omp_task);
  performance_analyzer->PipelineRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  // Verify results
  ASSERT_EQ(result_matrix.RowCount(), sparse_matrix_a.RowCount());
  ASSERT_EQ(result_matrix.ColumnCount(), sparse_matrix_b.ColumnCount());
  ASSERT_FALSE(result_matrix.task_data.empty());
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_task_run) {
  const uint32_t matrix_size = 400;
  const uint32_t non_zero_elements = 5000;
  const uint32_t num_runs = 10;

  // Create random sparse matrices
  auto sparse_matrix_a = CreateRandomSparseMatrix(matrix_size, non_zero_elements);
  auto sparse_matrix_b = CreateRandomSparseMatrix(matrix_size, non_zero_elements);

  // Create task data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sparse_matrix_a));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sparse_matrix_b));
  task_data->inputs_count.emplace_back(1);
  
  SparseMatrixFormat result_matrix;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_matrix));
  task_data->outputs_count.emplace_back(1);

  // Create Task
  auto omp_task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(task_data);

  // Create Performance attributes
  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = num_runs;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(omp_task);
  performance_analyzer->TaskRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  // Verify results
  ASSERT_EQ(result_matrix.RowCount(), sparse_matrix_a.RowCount());
  ASSERT_EQ(result_matrix.ColumnCount(), sparse_matrix_b.ColumnCount());
  ASSERT_FALSE(result_matrix.task_data.empty());
}
