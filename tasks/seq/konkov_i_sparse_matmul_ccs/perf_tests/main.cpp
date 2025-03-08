#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

TEST(konkov_i_SparseMatmulPerfTest_seq, test_pipeline_run) {
  constexpr int kSize = 5000;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<konkov_i_sparse_matmul_ccs::SparseMatmulTask>(task_data);

  std::vector<double> a_values(kSize, 1.0);
  std::vector<int> a_columns(kSize);
  std::vector<double> b_values(kSize, 1.0);
  std::vector<int> b_columns(kSize);

  for (int i = 0; i < kSize; i++) {
    a_columns[i] = i % kSize;
    b_columns[i] = i % kSize;
  }

  task->A_values = a_values;
  task->A_columns = a_columns;
  task->B_values = b_values;
  task->B_columns = b_columns;
  task->rowsA = kSize;
  task->colsA = kSize;
  task->rowsB = kSize;
  task->colsB = kSize;

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(konkov_i_SparseMatmulPerfTest_seq, test_task_run) {
  constexpr int kSize = 5000;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<konkov_i_sparse_matmul_ccs::SparseMatmulTask>(task_data);

  std::vector<double> a_values(kSize, 1.0);
  std::vector<int> a_columns(kSize);
  std::vector<double> b_values(kSize, 1.0);
  std::vector<int> b_columns(kSize);

  for (int i = 0; i < kSize; i++) {
    a_columns[i] = i % kSize;
    b_columns[i] = i % kSize;
  }

  task->A_values = a_values;
  task->A_columns = a_columns;
  task->B_values = b_values;
  task->B_columns = b_columns;
  task->rowsA = kSize;
  task->colsA = kSize;
  task->rowsB = kSize;
  task->colsB = kSize;

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
