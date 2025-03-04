#include <gtest/gtest.h>

#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

// Тест 1: Простой случай с разреженными матрицами
TEST(konkov_i_SparseMatmulTest, SimpleTest_seq) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  // Пример разреженных матриц A и B
  // A: 3x2, B: 2x2
  std::vector<double> A_values = {1.0, 2.0};  // Значения A
  std::vector<int> A_columns = {0, 1};        // Столбцы для A

  std::vector<double> B_values = {3.0, 4.0};  // Значения B
  std::vector<int> B_columns = {0, 1};        // Столбцы для B

  // Вводим в задачу
  task.A_values = A_values;
  task.A_columns = A_columns;
  task.B_values = B_values;
  task.B_columns = B_columns;
  task.rowsA = 3;
  task.colsA = 2;
  task.rowsB = 2;
  task.colsB = 2;

  // Запуск задачи
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}

// Тест 2: Проверка умножения пустых матриц
TEST(konkov_i_SparseMatmulTest, EmptyMatrixTest_seq) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  std::vector<double> A_values = {};  // Пустая матрица A
  std::vector<int> A_columns = {};    // Пустая матрица A

  std::vector<double> B_values = {};  // Пустая матрица B
  std::vector<int> B_columns = {};    // Пустая матрица B

  task.A_values = A_values;
  task.A_columns = A_columns;
  task.B_values = B_values;
  task.B_columns = B_columns;
  task.rowsA = 0;
  task.colsA = 0;
  task.rowsB = 0;
  task.colsB = 0;

  EXPECT_FALSE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}

// Тест 3: Сложный случай с большим количеством данных
TEST(konkov_i_SparseMatmulTest, ComplexTest_seq) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  // Пример разреженных матриц с большим числом элементов
  std::vector<double> A_values = {1.0, 0.0, 2.0, 0.0};  // Значения A
  std::vector<int> A_columns = {0, 1};                  // Столбцы для A

  std::vector<double> B_values = {3.0, 4.0};  // Значения B
  std::vector<int> B_columns = {0, 1};        // Столбцы для B

  task.A_values = A_values;
  task.A_columns = A_columns;
  task.B_values = B_values;
  task.B_columns = B_columns;
  task.rowsA = 3;
  task.colsA = 2;
  task.rowsB = 2;
  task.colsB = 2;

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}
