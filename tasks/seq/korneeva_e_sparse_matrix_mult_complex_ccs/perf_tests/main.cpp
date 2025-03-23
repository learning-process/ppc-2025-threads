#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_seq.hpp"

namespace korneeva_e_ccs = korneeva_e_sparse_matrix_mult_complex_ccs_seq;

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_pipeline_run) {
  constexpr int kRowsCols = 5000;
  constexpr int kVectorLength = kRowsCols;
  constexpr int kMaxNnzMatrix = 100;
  constexpr int kMaxNnzVector = 50;

  std::mt19937 gen(42);
  korneeva_e_ccs::SparseMatrixCCS matrix = korneeva_e_ccs::CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix, gen);
  korneeva_e_ccs::SparseMatrixCCS vector = korneeva_e_ccs::CreateRandomMatrix(kVectorLength, 1, kMaxNnzVector, gen);
  korneeva_e_ccs::SparseMatrixCCS result;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto test_task_sequential = std::make_shared<korneeva_e_ccs::SparseMatrixMultComplexCCS>(task_data_seq);

  ASSERT_TRUE(test_task_sequential->PreProcessingImpl());
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  ASSERT_TRUE(test_task_sequential->RunImpl());
  ASSERT_TRUE(test_task_sequential->PostProcessingImpl());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(result.rows, kRowsCols);
  ASSERT_EQ(result.cols, 1);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_task_run) {
  constexpr int kRowsCols = 5000;
  constexpr int kVectorLength = kRowsCols;
  constexpr int kMaxNnzMatrix = 100;
  constexpr int kMaxNnzVector = 50;

  std::mt19937 gen(42);
  korneeva_e_ccs::SparseMatrixCCS matrix = korneeva_e_ccs::CreateRandomMatrix(kRowsCols, kRowsCols, kMaxNnzMatrix, gen);
  korneeva_e_ccs::SparseMatrixCCS vector = korneeva_e_ccs::CreateRandomMatrix(kVectorLength, 1, kMaxNnzVector, gen);
  korneeva_e_ccs::SparseMatrixCCS result;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto test_task_sequential = std::make_shared<korneeva_e_ccs::SparseMatrixMultComplexCCS>(task_data_seq);

  ASSERT_TRUE(test_task_sequential->PreProcessingImpl());
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  ASSERT_TRUE(test_task_sequential->RunImpl());
  ASSERT_TRUE(test_task_sequential->PostProcessingImpl());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(result.rows, kRowsCols);
  ASSERT_EQ(result.cols, 1);
}