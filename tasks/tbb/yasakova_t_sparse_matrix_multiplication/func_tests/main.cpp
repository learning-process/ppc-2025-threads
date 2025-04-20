#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/yasakova_t_sparse_matrix_multiplication/include/ops_tbb.hpp"

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRealMatrices) {
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(3, 3);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(1, 0), 0);
  firstMatrix.InsertElement(0, ComplexNumber(2, 0), 2);
  firstMatrix.InsertElement(1, ComplexNumber(3, 0), 1);
  firstMatrix.InsertElement(2, ComplexNumber(4, 0), 0);
  firstMatrix.InsertElement(2, ComplexNumber(5, 0), 1);

  secondMatrix.InsertElement(0, ComplexNumber(6, 0), 1);
  secondMatrix.InsertElement(1, ComplexNumber(7, 0), 0);
  secondMatrix.InsertElement(2, ComplexNumber(8, 0), 2);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(6, 0), 1);
  expectedResult.InsertElement(0, ComplexNumber(16, 0), 2);
  expectedResult.InsertElement(1, ComplexNumber(21, 0), 0);
  expectedResult.InsertElement(2, ComplexNumber(24, 0), 1);
  expectedResult.InsertElement(2, ComplexNumber(35, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, InvalidMatrixDimensions) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(5, 3);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(1, 0), 0);
  firstMatrix.InsertElement(0, ComplexNumber(2, 0), 2);
  firstMatrix.InsertElement(1, ComplexNumber(3, 0), 1);
  firstMatrix.InsertElement(2, ComplexNumber(4, 0), 0);
  firstMatrix.InsertElement(2, ComplexNumber(5, 0), 1);

  secondMatrix.InsertElement(0, ComplexNumber(6, 0), 1);
  secondMatrix.InsertElement(1, ComplexNumber(7, 0), 0);
  secondMatrix.InsertElement(2, ComplexNumber(8, 0), 2);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyComplexMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(3, 3);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(1, 1), 0);
  firstMatrix.InsertElement(0, ComplexNumber(2, 2), 2);
  firstMatrix.InsertElement(1, ComplexNumber(3, 3), 1);
  firstMatrix.InsertElement(2, ComplexNumber(4, 4), 0);
  firstMatrix.InsertElement(2, ComplexNumber(5, 5), 1);

  secondMatrix.InsertElement(0, ComplexNumber(6, 6), 1);
  secondMatrix.InsertElement(1, ComplexNumber(7, 7), 0);
  secondMatrix.InsertElement(2, ComplexNumber(8, 8), 2);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(0, 12), 1);
  expectedResult.InsertElement(0, ComplexNumber(0, 32), 2);
  expectedResult.InsertElement(1, ComplexNumber(0, 42), 0);
  expectedResult.InsertElement(2, ComplexNumber(0, 48), 1);
  expectedResult.InsertElement(2, ComplexNumber(0, 70), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRectangularMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(2, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(3, 4);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(2, 4);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(1, 0), 1);
  firstMatrix.InsertElement(0, ComplexNumber(2, 0), 2);
  firstMatrix.InsertElement(1, ComplexNumber(3, 0), 1);

  secondMatrix.InsertElement(0, ComplexNumber(3, 0), 2);
  secondMatrix.InsertElement(1, ComplexNumber(5, 0), 0);
  secondMatrix.InsertElement(1, ComplexNumber(4, 0), 3);
  secondMatrix.InsertElement(2, ComplexNumber(7, 0), 0);
  secondMatrix.InsertElement(2, ComplexNumber(8, 0), 1);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(19, 0), 0);
  expectedResult.InsertElement(0, ComplexNumber(4, 0), 3);
  expectedResult.InsertElement(0, ComplexNumber(16, 0), 1);
  expectedResult.InsertElement(1, ComplexNumber(15, 0), 0);
  expectedResult.InsertElement(1, ComplexNumber(12, 0), 3);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyMatricesWithNegativeElements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(2, 2);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(-1, -1), 0);
  firstMatrix.InsertElement(1, ComplexNumber(3, 3), 1);

  secondMatrix.InsertElement(0, ComplexNumber(6, 6), 1);
  secondMatrix.InsertElement(1, ComplexNumber(-7, -7), 0);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(0, -12), 1);
  expectedResult.InsertElement(1, ComplexNumber(0, -42), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyMatricesWithDoubleElements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(2, 2);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(-1.7, -1.5), 0);
  firstMatrix.InsertElement(1, ComplexNumber(3.7, 3.1), 1);

  secondMatrix.InsertElement(0, ComplexNumber(6.3, 6.1), 1);
  secondMatrix.InsertElement(1, ComplexNumber(-7.4, -7.7), 0);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(-1.56, -19.82), 1);
  expectedResult.InsertElement(1, ComplexNumber(-3.51, -51.43), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRowByColumnMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(1, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(3, 1);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(1, 1);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(-1, 0), 0);
  firstMatrix.InsertElement(0, ComplexNumber(-2, 0), 1);
  firstMatrix.InsertElement(0, ComplexNumber(-3, 0), 2);

  secondMatrix.InsertElement(0, ComplexNumber(1, 0), 0);
  secondMatrix.InsertElement(1, ComplexNumber(2, 0), 0);
  secondMatrix.InsertElement(2, ComplexNumber(3, 0), 0);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(-14, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyDiagonalMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(3, 3);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(-1, 0), 0);
  firstMatrix.InsertElement(1, ComplexNumber(-2, 0), 1);
  firstMatrix.InsertElement(2, ComplexNumber(-3, 0), 2);

  secondMatrix.InsertElement(0, ComplexNumber(1, 0), 0);
  secondMatrix.InsertElement(1, ComplexNumber(2, 0), 1);
  secondMatrix.InsertElement(2, ComplexNumber(3, 0), 2);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(-1, 0), 0);
  expectedResult.InsertElement(1, ComplexNumber(-4, 0), 1);
  expectedResult.InsertElement(2, ComplexNumber(-9, 0), 2);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyImaginaryMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix firstMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix secondMatrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expectedResult(3, 3);
  std::vector<ComplexNumber> inputData = {};
  std::vector<ComplexNumber> firstMatrixData;
  std::vector<ComplexNumber> secondMatrixData;
  std::vector<ComplexNumber> outputData(firstMatrix.columnCount * secondMatrix.rowCount * 100, 0);

  firstMatrix.InsertElement(0, ComplexNumber(0, 1), 0);
  firstMatrix.InsertElement(0, ComplexNumber(0, 2), 2);
  firstMatrix.InsertElement(1, ComplexNumber(0, 3), 1);
  firstMatrix.InsertElement(2, ComplexNumber(0, 4), 0);
  firstMatrix.InsertElement(2, ComplexNumber(0, 5), 1);

  secondMatrix.InsertElement(0, ComplexNumber(0, 6), 1);
  secondMatrix.InsertElement(1, ComplexNumber(0, 7), 0);
  secondMatrix.InsertElement(2, ComplexNumber(0, 8), 2);
  firstMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(firstMatrix);
  secondMatrixData = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(secondMatrix);
  inputData.reserve(firstMatrixData.size() + secondMatrixData.size());
  for (unsigned int i = 0; i < firstMatrixData.size(); i++) {
    inputData.emplace_back(firstMatrixData[i]);
  }
  for (unsigned int i = 0; i < secondMatrixData.size(); i++) {
    inputData.emplace_back(secondMatrixData[i]);
  }
  expectedResult.InsertElement(0, ComplexNumber(-6, 0), 1);
  expectedResult.InsertElement(0, ComplexNumber(-16, 0), 2);
  expectedResult.InsertElement(1, ComplexNumber(-21, 0), 0);
  expectedResult.InsertElement(2, ComplexNumber(-24, 0), 1);
  expectedResult.InsertElement(2, ComplexNumber(-35, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  task_data_tbb->inputs_count.emplace_back(inputData.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  task_data_tbb->outputs_count.emplace_back(outputData.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actualResult =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(outputData);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actualResult, expectedResult));
}