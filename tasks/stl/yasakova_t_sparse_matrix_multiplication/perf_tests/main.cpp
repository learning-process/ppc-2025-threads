#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/yasakova_t_sparse_matrix_multiplication/include/ops_stl.hpp"

namespace {
yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage GenMatrix(
    unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
    unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value);
yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage GenMatrix(
    unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
    unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value) {
  if (left_border_row > right_border_row || left_border_col > right_border_col || right_border_row > num_rows ||
      right_border_col > num_cols || min_value > max_value) {
    throw("ERROR!");
  }
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix((int)num_rows, (int)num_cols);
  for (unsigned int i = left_border_row; i < right_border_row; i++) {
    for (unsigned int j = left_border_col; j < right_border_col; j++) {
      left_matrix.InsertElement((int)i, Complex(min_value + (rand() % max_value), min_value + (rand() % max_value)), (int)j);
    }
  }
  return left_matrix;
}
}  // namespace
TEST(yasakova_t_sparse_matrix_multiplication_task_stl, test_pipeline_run) {
  srand(time(nullptr));
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(400, 400);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(400, 400);
  std::vector<Complex> input_data = {};
  std::vector<Complex> left_matrix_data;
  std::vector<Complex> right_matrix_data;
  std::vector<Complex> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  right_matrix = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  auto multiplication_task = std::make_shared<yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask>(task_data_stl);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(multiplication_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
}

TEST(yasakova_t_sparse_matrix_multiplication_task_stl, test_task_run) {
  srand(time(nullptr));
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(400, 400);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(400, 400);
  std::vector<Complex> input_data = {};
  std::vector<Complex> left_matrix_data;
  std::vector<Complex> right_matrix_data;
  std::vector<Complex> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  right_matrix = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }

  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  auto multiplication_task = std::make_shared<yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask>(task_data_stl);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(multiplication_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
}