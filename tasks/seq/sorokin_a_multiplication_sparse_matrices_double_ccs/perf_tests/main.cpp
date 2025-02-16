#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_seq.hpp"

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_pipeline_run) {
  int M = 20000;
  int K = 20000;
  int N = 20000;

  std::vector<double> A_values(20000, 1);
  std::vector<double> A_row_indices(20000);
  for (size_t i = 0; i < 20000; i++) {
    A_row_indices[i] = i;
  }
  std::vector<double> A_col_ptr(20001);
  for (size_t i = 0; i <= 20000; i++) {
    A_col_ptr[i] = i;
  }
  std::vector<double> B_values(20000, 1);
  std::vector<double> B_row_indices(20000);
  for (size_t i = 0; i < 20000; i++) {
    B_row_indices[i] = 19999 - i;
  }
  std::vector<double> B_col_ptr(20001);
  for (size_t i = 0; i <= 20000; i++) {
    B_col_ptr[i] = i;
  }

  std::vector<double> C_values(100000);
  std::vector<double> C_row_indices(100000);
  std::vector<double> C_col_ptr(100000);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential>(task_data_seq);

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
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_task_run) {
  int M = 20000;
  int K = 20000;
  int N = 20000;

  std::vector<double> A_values(20000, 1);
  std::vector<double> A_row_indices(20000);
  for (size_t i = 0; i < 20000; i++) {
    A_row_indices[i] = i;
  }
  std::vector<double> A_col_ptr(20001);
  for (size_t i = 0; i <= 20000; i++) {
    A_col_ptr[i] = i;
  }
  std::vector<double> B_values(20000, 1);
  std::vector<double> B_row_indices(20000);
  for (size_t i = 0; i < 20000; i++) {
    B_row_indices[i] = 19999 - i;
  }
  std::vector<double> B_col_ptr(20001);
  for (size_t i = 0; i <= 20000; i++) {
    B_col_ptr[i] = i;
  }

  std::vector<double> C_values(100000);
  std::vector<double> C_row_indices(100000);
  std::vector<double> C_col_ptr(100000);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential>(task_data_seq);

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
}
