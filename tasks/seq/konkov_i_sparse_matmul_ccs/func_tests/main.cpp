#include <gtest/gtest.h>

#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

TEST(konkov_i_SparseMatmulTest_seq, SimpleTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  std::vector<double> A_values = {1.0, 2.0};
  std::vector<int> A_columns = {0, 1};

  std::vector<double> B_values = {3.0, 4.0};
  std::vector<int> B_columns = {0, 1};

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

TEST(konkov_i_SparseMatmulTest_seq, EmptyMatrixTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  std::vector<double> A_values = {};
  std::vector<int> A_columns = {};

  std::vector<double> B_values = {};
  std::vector<int> B_columns = {};

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

TEST(konkov_i_SparseMatmulTest_seq, ComplexTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  std::vector<double> A_values = {1.0, 0.0, 2.0, 0.0};
  std::vector<int> A_columns = {0, 1};

  std::vector<double> B_values = {3.0, 4.0};
  std::vector<int> B_columns = {0, 1};

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
