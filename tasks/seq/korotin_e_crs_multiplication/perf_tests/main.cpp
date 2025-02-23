#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/korotin_e_crs_multiplication/include/ops_seq.hpp"

TEST(korotin_e_crs_multiplication_seq, test_pipeline_run) {
  const unsigned int N = 100;

  std::vector<double> A_val(N * N, 1), B_val(N * N, 1), C_val(N * N, N);
  std::vector<unsigned int> A_rI(N + 1, 0), A_col(N * N), B_rI(N + 1, 0), B_col(N * N), C_rI(N + 1, 0), C_col(N * N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_col[i * N + j] = j;
      B_col[i * N + j] = j;
      C_col[i * N + j] = j;
    }
    A_rI[i + 1] = N * (i + 1);
    B_rI[i + 1] = N * (i + 1);
    C_rI[i + 1] = N * (i + 1);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
  task_data_seq->inputs_count.emplace_back(A_rI.size());
  task_data_seq->inputs_count.emplace_back(A_col.size());
  task_data_seq->inputs_count.emplace_back(A_val.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
  task_data_seq->inputs_count.emplace_back(B_rI.size());
  task_data_seq->inputs_count.emplace_back(B_col.size());
  task_data_seq->inputs_count.emplace_back(B_val.size());

  std::vector<unsigned int> out_rI(A_rI.size(), 0), out_col(N * N);
  std::vector<double> out_val(N * N);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_rI.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_seq->outputs_count.emplace_back(out_rI.size());

  // Create Task
  auto test_task_sequential = std::make_shared<korotin_e_crs_multiplication_seq::CrsMultiplicationSequential>(task_data_seq);

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

  ASSERT_EQ(C_rI, out_rI);
  ASSERT_EQ(C_col, out_col);
  ASSERT_EQ(C_val, out_val);
}

TEST(korotin_e_crs_multiplication_seq, test_task_run) {
  const unsigned int N = 100;

  std::vector<double> A_val(N * N, 1), B_val(N * N, 1), C_val(N * N, N);
  std::vector<unsigned int> A_rI(N + 1, 0), A_col(N * N), B_rI(N + 1, 0), B_col(N * N), C_rI(N + 1, 0), C_col(N * N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_col[i * N + j] = j;
      B_col[i * N + j] = j;
      C_col[i * N + j] = j;
    }
    A_rI[i + 1] = N * (i + 1);
    B_rI[i + 1] = N * (i + 1);
    C_rI[i + 1] = N * (i + 1);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
  task_data_seq->inputs_count.emplace_back(A_rI.size());
  task_data_seq->inputs_count.emplace_back(A_col.size());
  task_data_seq->inputs_count.emplace_back(A_val.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
  task_data_seq->inputs_count.emplace_back(B_rI.size());
  task_data_seq->inputs_count.emplace_back(B_col.size());
  task_data_seq->inputs_count.emplace_back(B_val.size());

  std::vector<unsigned int> out_rI(A_rI.size(), 0), out_col(N * N);
  std::vector<double> out_val(N * N);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_rI.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_seq->outputs_count.emplace_back(out_rI.size());

  // Create Task
  auto test_task_sequential = std::make_shared<korotin_e_crs_multiplication_seq::CrsMultiplicationSequential>(task_data_seq);

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

  ASSERT_EQ(C_rI, out_rI);
  ASSERT_EQ(C_col, out_col);
  ASSERT_EQ(C_val, out_val);
}
