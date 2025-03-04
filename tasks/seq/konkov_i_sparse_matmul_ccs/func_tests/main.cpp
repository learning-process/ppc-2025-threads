#include <gtest/gtest.h>

#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

// ���� 1: ������� ������ � ������������ ���������
TEST(konkov_i_SparseMatmulTest, SimpleTest_seq) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  // ������ ����������� ������ A � B
  // A: 3x2, B: 2x2
  std::vector<double> A_values = {1.0, 2.0};  // �������� A
  std::vector<int> A_columns = {0, 1};        // ������� ��� A

  std::vector<double> B_values = {3.0, 4.0};  // �������� B
  std::vector<int> B_columns = {0, 1};        // ������� ��� B

  // ������ � ������
  task.A_values = A_values;
  task.A_columns = A_columns;
  task.B_values = B_values;
  task.B_columns = B_columns;
  task.rowsA = 3;
  task.colsA = 2;
  task.rowsB = 2;
  task.colsB = 2;

  // ������ ������
  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}

// ���� 2: �������� ��������� ������ ������
TEST(konkov_i_SparseMatmulTest, EmptyMatrixTest_seq) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  std::vector<double> A_values = {};  // ������ ������� A
  std::vector<int> A_columns = {};    // ������ ������� A

  std::vector<double> B_values = {};  // ������ ������� B
  std::vector<int> B_columns = {};    // ������ ������� B

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

// ���� 3: ������� ������ � ������� ����������� ������
TEST(konkov_i_SparseMatmulTest, ComplexTest_seq) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  // ������ ����������� ������ � ������� ������ ���������
  std::vector<double> A_values = {1.0, 0.0, 2.0, 0.0};  // �������� A
  std::vector<int> A_columns = {0, 1};                  // ������� ��� A

  std::vector<double> B_values = {3.0, 4.0};  // �������� B
  std::vector<int> B_columns = {0, 1};        // ������� ��� B

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
