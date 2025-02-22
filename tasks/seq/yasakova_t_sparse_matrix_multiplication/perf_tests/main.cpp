#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_pipeline_run) {
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparseMatrixA(400, true, 400);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparseMatrixB(400, true, 400);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vectorA;
  std::vector<Complex> vectorB;
  std::vector<Complex> resultVector(sparseMatrixA.columnCount * sparseMatrixB.rowCount * 100, 0);

  for (unsigned int row = 0; row < 150; row++) {
    for (unsigned int col = 0; col < 150; col++) {
      sparseMatrixA.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)), static_cast<int>(col));
    }
  }
  for (unsigned int row = 50; row < 140; row++) {
    for (unsigned int col = 50; col < 150; col++) {
      sparseMatrixB.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)), static_cast<int>(col));
    }
  }
  vectorA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparseMatrixA);
  vectorB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparseMatrixB);
  inputData.reserve(vectorA.size() + vectorB.size());
  inputData.insert(inputData.end(), vectorA.begin(), vectorA.end());
  inputData.insert(inputData.end(), vectorB.begin(), vectorB.end());

  // Initialize task data structure
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskData->outputs_count.emplace_back(resultVector.size());

  // Create Task
  auto sequentialTask = std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(taskData);

  // Create Performance attributes
  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;
  const auto startTime = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&] {
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performanceResults = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(sequentialTask);
  performanceAnalyzer->PipelineRun(performanceAttributes, performanceResults);
  ppc::core::Perf::PrintPerfStatistic(performanceResults);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS finalResult =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(resultVector);
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_task_run) {
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparseMatrixA(400, true, 400);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparseMatrixB(400, true, 400);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vectorA;
  std::vector<Complex> vectorB;
  std::vector<Complex> resultVector(sparseMatrixA.columnCount * sparseMatrixB.rowCount * 100, 0);

  for (unsigned int row = 0; row < 150; row++) {
    for (unsigned int col = 0; col < 150; col++) {
      sparseMatrixA.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)), static_cast<int>(col));
    }
  }
  for (unsigned int row = 50; row < 140; row++) {
    for (unsigned int col = 50; col < 150; col++) {
      sparseMatrixB.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)), static_cast<int>(col));
    }
  }
  vectorA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparseMatrixA);
  vectorB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparseMatrixB);
  inputData.reserve(vectorA.size() + vectorB.size());
  inputData.insert(inputData.end(), vectorA.begin(), vectorA.end());
  inputData.insert(inputData.end(), vectorB.begin(), vectorB.end());

  // Initialize task data structure
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskData->outputs_count.emplace_back(resultVector.size());

  // Create Task
  auto sequentialTask = std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(taskData);

  // Create Performance attributes
  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;
  const auto startTime = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&] {
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performanceResults = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(sequentialTask);
  performanceAnalyzer->TaskRun(performanceAttributes, performanceResults);
  ppc::core::Perf::PrintPerfStatistic(performanceResults);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS finalResult =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(resultVector);
}