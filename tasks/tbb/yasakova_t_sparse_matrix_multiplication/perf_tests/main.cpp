#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/yasakova_t_sparse_matrix_multiplication/include/ops_tbb.hpp"

namespace {
yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix GenMatrix(
    unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
    unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value) {
  if (left_border_row > right_border_row || left_border_col > right_border_col || right_border_row > num_rows ||
      right_border_col > num_cols || min_value > max_value) {
    throw("ERROR!");
  }
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix((int)num_rows, (int)num_cols);
  for (unsigned int i = left_border_row; i < right_border_row; i++) {
    for (unsigned int j = left_border_col; j < right_border_col; j++) {
      first_matrix.InsertElement(
          (int)i, ComplexNumber(min_value + (rand() % max_value), min_value + (rand() % max_value)), (int)j);
    }
  }
  return first_matrix;
}
}  // namespace
TEST(yasakova_t_sparse_matrix_multiplication_tbb, test_pipeline_run) {
  srand(time(nullptr));
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(400, 400);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(400, 400);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  second_matrix = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  auto test_task_tbb = std::make_shared<yasakova_t_sparse_matrix_multiplication::TestTaskTBB>(task_data_tbb);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
}

TEST(yasakova_t_sparse_matrix_multiplication_tbb, test_task_run) {
  srand(time(nullptr));
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(400, 400);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(400, 400);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  second_matrix = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  auto test_task_tbb = std::make_shared<yasakova_t_sparse_matrix_multiplication::TestTaskTBB>(task_data_tbb);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
}