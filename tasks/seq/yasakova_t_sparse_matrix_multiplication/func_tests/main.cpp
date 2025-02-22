#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_real_matrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(3, true, 3);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  matA.InsertElement(0, Complex(1, 0), 0);
  matA.InsertElement(0, Complex(2, 0), 2);
  matA.InsertElement(1, Complex(3, 0), 1);
  matA.InsertElement(2, Complex(4, 0), 0);
  matA.InsertElement(2, Complex(5, 0), 1);

  matB.InsertElement(0, Complex(6, 0), 1);
  matB.InsertElement(1, Complex(7, 0), 0);
  matB.InsertElement(2, Complex(8, 0), 2);
  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }
  expectedResult.InsertElement(0, Complex(6, 0), 1);
  expectedResult.InsertElement(0, Complex(16, 0), 2);
  expectedResult.InsertElement(1, Complex(21, 0), 0);
  expectedResult.InsertElement(2, Complex(24, 0), 1);
  expectedResult.InsertElement(2, Complex(35, 0), 0);

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_imaginary_parts) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(3, true, 3);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  matA.InsertElement(0, Complex(1, 1), 0);
  matA.InsertElement(0, Complex(2, 2), 2);
  matA.InsertElement(1, Complex(3, 3), 1);
  matA.InsertElement(2, Complex(4, 4), 0);
  matA.InsertElement(2, Complex(5, 5), 1);

  matB.InsertElement(0, Complex(6, 6), 1);
  matB.InsertElement(1, Complex(7, 7), 0);
  matB.InsertElement(2, Complex(8, 8), 2);
  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }
  expectedResult.InsertElement(0, Complex(0, 12), 1);
  expectedResult.InsertElement(0, Complex(0, 32), 2);
  expectedResult.InsertElement(1, Complex(0, 42), 0);
  expectedResult.InsertElement(2, Complex(0, 48), 1);
  expectedResult.InsertElement(2, Complex(0, 70), 0);

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_rectangular_matrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(2, false, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(3, false, 4);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(2, false, 4);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  matA.InsertElement(0, Complex(1, 0), 1);
  matA.InsertElement(0, Complex(2, 0), 2);
  matA.InsertElement(1, Complex(3, 0), 1);

  matB.InsertElement(0, Complex(3, 0), 2);
  matB.InsertElement(1, Complex(5, 0), 0);
  matB.InsertElement(1, Complex(4, 0), 3);
  matB.InsertElement(2, Complex(7, 0), 0);
  matB.InsertElement(2, Complex(8, 0), 1);
  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }
  expectedResult.InsertElement(0, Complex(19, 0), 0);
  expectedResult.InsertElement(0, Complex(4, 0), 3);
  expectedResult.InsertElement(0, Complex(16, 0), 1);
  expectedResult.InsertElement(1, Complex(15, 0), 0);
  expectedResult.InsertElement(1, Complex(12, 0), 3);

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_negative_elements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(2, true, 2);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  matA.InsertElement(0, Complex(-1, -1), 0);
  matA.InsertElement(1, Complex(3, 3), 1);

  matB.InsertElement(0, Complex(6, 6), 1);
  matB.InsertElement(1, Complex(-7, -7), 0);
  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }
  expectedResult.InsertElement(0, Complex(0, -12), 1);
  expectedResult.InsertElement(1, Complex(0, -42), 0);

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_zero_elements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(2, true, 2);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(2, true, 2);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  matA.InsertElement(0, Complex(0, 0), 0);
  matA.InsertElement(1, Complex(0, 0), 1);

  matB.InsertElement(0, Complex(0, 0), 1);
  matB.InsertElement(1, Complex(0, 0), 0);
  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }
  expectedResult.InsertElement(0, Complex(0, 0), 1);
  expectedResult.InsertElement(1, Complex(0, 0), 0);

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_matrices_with_different_dimensions) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(5, false, 3);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  matA.InsertElement(0, Complex(1, 0), 0);
  matA.InsertElement(0, Complex(2, 0), 2);
  matA.InsertElement(1, Complex(3, 0), 1);
  matA.InsertElement(2, Complex(4, 0), 0);
  matA.InsertElement(2, Complex(5, 0), 1);

  matB.InsertElement(0, Complex(6, 0), 1);
  matB.InsertElement(1, Complex(7, 0), 0);
  matB.InsertElement(2, Complex(8, 0), 2);
  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), false);
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_zero_matrix) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(3, true, 3);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  // matA is zero matrix
  matB.InsertElement(0, Complex(1, 0), 0);
  matB.InsertElement(1, Complex(2, 0), 1);
  matB.InsertElement(2, Complex(3, 0), 2);

  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }

  // Expected result is zero matrix
  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_multiply_identity_matrix) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matA(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS matB(3, true, 3);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expectedResult(3, true, 3);
  std::vector<Complex> inputData = {};
  std::vector<Complex> vecA;
  std::vector<Complex> vecB;
  std::vector<Complex> outputData(matA.columnCount * matB.rowCount * 100, 0);

  // matA is identity matrix
  matA.InsertElement(0, Complex(1, 0), 0);
  matA.InsertElement(1, Complex(1, 0), 1);
  matA.InsertElement(2, Complex(1, 0), 2);

  matB.InsertElement(0, Complex(2, 0), 0);
  matB.InsertElement(1, Complex(3, 0), 1);
  matB.InsertElement(2, Complex(4, 0), 2);

  // Expected result is matB
  expectedResult.InsertElement(0, Complex(2, 0), 0);
  expectedResult.InsertElement(1, Complex(3, 0), 1);
  expectedResult.InsertElement(2, Complex(4, 0), 2);

  vecA = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matA);
  vecB = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(matB);
  inputData.reserve(vecA.size() + vecB.size());
  for (unsigned int i = 0; i < vecA.size(); i++) {
    inputData.emplace_back(vecA[i]);
  }
  for (unsigned int i = 0; i < vecB.size(); i++) {
    inputData.emplace_back(vecB[i]);
  }

  // Create task_data
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.emplace_back(inputData.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest testTask(taskData);
  ASSERT_EQ(testTask.Validation(), true);
  testTask.PreProcessing();
  testTask.Run();
  testTask.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(result, expectedResult));
}