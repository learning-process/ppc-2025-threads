#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

TEST(konkov_i_SparseMatmulTest_seq, SimpleTest) {
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  std::vector<double> a_values = {1.0, 2.0};
  std::vector<int> a_columns = {0, 1};

  std::vector<double> b_values = {3.0, 4.0};
  std::vector<int> b_columns = {0, 1};

  task.A_values = a_values;
  task.A_columns = a_columns;
  task.B_values = b_values;
  task.B_columns = b_columns;
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

  std::vector<double> a_values = {};
  std::vector<int> a_columns = {};

  std::vector<double> b_values = {};
  std::vector<int> b_columns = {};

  task.A_values = a_values;
  task.A_columns = a_columns;
  task.B_values = b_values;
  task.B_columns = b_columns;
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

  std::vector<double> a_values = {1.0, 0.0, 2.0, 0.0};
  std::vector<int> a_columns = {0, 1};

  std::vector<double> b_values = {3.0, 4.0};
  std::vector<int> b_columns = {0, 1};

  task.A_values = a_values;
  task.A_columns = a_columns;
  task.B_values = b_values;
  task.B_columns = b_columns;
  task.rowsA = 3;
  task.colsA = 2;
  task.rowsB = 2;
  task.colsB = 2;

  EXPECT_TRUE(task.ValidationImpl());
  EXPECT_TRUE(task.PreProcessingImpl());
  EXPECT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());
}
