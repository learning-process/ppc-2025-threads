#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/yasakova_t_sparse_matrix_multiplication/include/ops_tbb.hpp"

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRealMatrices) {
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  first_matrix.InsertElement(0, ComplexNumber(2, 0), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 0), 1);
  first_matrix.InsertElement(2, ComplexNumber(4, 0), 0);
  first_matrix.InsertElement(2, ComplexNumber(5, 0), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 0), 1);
  second_matrix.InsertElement(1, ComplexNumber(7, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 0), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(6, 0), 1);
  expected_result.InsertElement(0, ComplexNumber(16, 0), 2);
  expected_result.InsertElement(1, ComplexNumber(21, 0), 0);
  expected_result.InsertElement(2, ComplexNumber(24, 0), 1);
  expected_result.InsertElement(2, ComplexNumber(35, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, InvalidMatrixDimensions) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(5, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  first_matrix.InsertElement(0, ComplexNumber(2, 0), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 0), 1);
  first_matrix.InsertElement(2, ComplexNumber(4, 0), 0);
  first_matrix.InsertElement(2, ComplexNumber(5, 0), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 0), 1);
  second_matrix.InsertElement(1, ComplexNumber(7, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 0), 2);
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
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), false);
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyComplexMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 1), 0);
  first_matrix.InsertElement(0, ComplexNumber(2, 2), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 3), 1);
  first_matrix.InsertElement(2, ComplexNumber(4, 4), 0);
  first_matrix.InsertElement(2, ComplexNumber(5, 5), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 6), 1);
  second_matrix.InsertElement(1, ComplexNumber(7, 7), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 8), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(0, 12), 1);
  expected_result.InsertElement(0, ComplexNumber(0, 32), 2);
  expected_result.InsertElement(1, ComplexNumber(0, 42), 0);
  expected_result.InsertElement(2, ComplexNumber(0, 48), 1);
  expected_result.InsertElement(2, ComplexNumber(0, 70), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRectangularMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(2, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 4);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(2, 4);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(1, 0), 1);
  first_matrix.InsertElement(0, ComplexNumber(2, 0), 2);
  first_matrix.InsertElement(1, ComplexNumber(3, 0), 1);

  second_matrix.InsertElement(0, ComplexNumber(3, 0), 2);
  second_matrix.InsertElement(1, ComplexNumber(5, 0), 0);
  second_matrix.InsertElement(1, ComplexNumber(4, 0), 3);
  second_matrix.InsertElement(2, ComplexNumber(7, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(8, 0), 1);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(19, 0), 0);
  expected_result.InsertElement(0, ComplexNumber(4, 0), 3);
  expected_result.InsertElement(0, ComplexNumber(16, 0), 1);
  expected_result.InsertElement(1, ComplexNumber(15, 0), 0);
  expected_result.InsertElement(1, ComplexNumber(12, 0), 3);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyMatricesWithNegativeElements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(2, 2);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1, -1), 0);
  first_matrix.InsertElement(1, ComplexNumber(3, 3), 1);

  second_matrix.InsertElement(0, ComplexNumber(6, 6), 1);
  second_matrix.InsertElement(1, ComplexNumber(-7, -7), 0);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(0, -12), 1);
  expected_result.InsertElement(1, ComplexNumber(0, -42), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyMatricesWithDoubleElements) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(2, 2);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1.7, -1.5), 0);
  first_matrix.InsertElement(1, ComplexNumber(3.7, 3.1), 1);

  second_matrix.InsertElement(0, ComplexNumber(6.3, 6.1), 1);
  second_matrix.InsertElement(1, ComplexNumber(-7.4, -7.7), 0);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-1.56, -19.82), 1);
  expected_result.InsertElement(1, ComplexNumber(-3.51, -51.43), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyRowByColumnMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(1, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 1);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(1, 1);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1, 0), 0);
  first_matrix.InsertElement(0, ComplexNumber(-2, 0), 1);
  first_matrix.InsertElement(0, ComplexNumber(-3, 0), 2);

  second_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  second_matrix.InsertElement(1, ComplexNumber(2, 0), 0);
  second_matrix.InsertElement(2, ComplexNumber(3, 0), 0);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-14, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyDiagonalMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(-1, 0), 0);
  first_matrix.InsertElement(1, ComplexNumber(-2, 0), 1);
  first_matrix.InsertElement(2, ComplexNumber(-3, 0), 2);

  second_matrix.InsertElement(0, ComplexNumber(1, 0), 0);
  second_matrix.InsertElement(1, ComplexNumber(2, 0), 1);
  second_matrix.InsertElement(2, ComplexNumber(3, 0), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-1, 0), 0);
  expected_result.InsertElement(1, ComplexNumber(-4, 0), 1);
  expected_result.InsertElement(2, ComplexNumber(-9, 0), 2);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyImaginaryMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(3, 3);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(0, 1), 0);
  first_matrix.InsertElement(0, ComplexNumber(0, 2), 2);
  first_matrix.InsertElement(1, ComplexNumber(0, 3), 1);
  first_matrix.InsertElement(2, ComplexNumber(0, 4), 0);
  first_matrix.InsertElement(2, ComplexNumber(0, 5), 1);

  second_matrix.InsertElement(0, ComplexNumber(0, 6), 1);
  second_matrix.InsertElement(1, ComplexNumber(0, 7), 0);
  second_matrix.InsertElement(2, ComplexNumber(0, 8), 2);
  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }
  expected_result.InsertElement(0, ComplexNumber(-6, 0), 1);
  expected_result.InsertElement(0, ComplexNumber(-16, 0), 2);
  expected_result.InsertElement(1, ComplexNumber(-21, 0), 0);
  expected_result.InsertElement(2, ComplexNumber(-24, 0), 1);
  expected_result.InsertElement(2, ComplexNumber(-35, 0), 0);
  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplyEmptyMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(0, 0);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(0, 0);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(0, 0);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(1, 0);

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
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication, MultiplySingleElementMatrices) {
  // Create data
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix(1, 1);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix second_matrix(1, 1);
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix expected_result(1, 1);
  std::vector<ComplexNumber> input_data = {};
  std::vector<ComplexNumber> first_matrix_data;
  std::vector<ComplexNumber> second_matrix_data;
  std::vector<ComplexNumber> output_data(first_matrix.columnCount * second_matrix.rowCount * 100, 0);

  first_matrix.InsertElement(0, ComplexNumber(3.5, 2.1), 0);
  second_matrix.InsertElement(0, ComplexNumber(1.2, 0.7), 0);

  first_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(first_matrix);
  second_matrix_data = yasakova_t_sparse_matrix_multiplication::ConvertMatrixToVector(second_matrix);
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  for (unsigned int i = 0; i < first_matrix_data.size(); i++) {
    input_data.emplace_back(first_matrix_data[i]);
  }
  for (unsigned int i = 0; i < second_matrix_data.size(); i++) {
    input_data.emplace_back(second_matrix_data[i]);
  }

  // (3.5+2.1i)*(1.2+0.7i) = (3.5*1.2 - 2.1*0.7) + (3.5*0.7 + 2.1*1.2)i
  // = (4.2 - 1.47) + (2.45 + 2.52)i = 2.73 + 4.97i
  expected_result.InsertElement(0, ComplexNumber(2.73, 4.97), 0);

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_tbb->inputs_count.emplace_back(input_data.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_tbb->outputs_count.emplace_back(output_data.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix actual_result =
      yasakova_t_sparse_matrix_multiplication::ConvertVectorToMatrix(output_data);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication::CompareMatrices(actual_result, expected_result));
}