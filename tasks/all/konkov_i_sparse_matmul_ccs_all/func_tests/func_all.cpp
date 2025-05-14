#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <memory>
#include <vector>

#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(konkov_i_SparseMatmulTest_all, SimpleTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs_all::SparseMatmulTask task(task_data);

  if (world.rank() == 0) {
    task.A_values = {5.0, 7.0, 9.0};
    task.A_row_indices = {0, 1, 2};
    task.A_col_ptr = {0, 1, 2, 3};
    task.rowsA = 3;
    task.colsA = 3;

    task.B_values = {3.0, 4.0, 2.0};
    task.B_row_indices = {0, 1, 2};
    task.B_col_ptr = {0, 1, 2, 3};
    task.rowsB = 3;
    task.colsB = 3;
  }

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (world.rank() == 0) {
    std::vector<double> expected_values = {15.0, 28.0, 18.0};
    EXPECT_EQ(task.C_values, expected_values);
  }
}

TEST(konkov_i_SparseMatmulTest_all, EmptyMatrixTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs_all::SparseMatmulTask task(task_data);

  if (world.rank() == 0) {
    task.A_col_ptr = {0};
    task.B_col_ptr = {0};
    task.rowsA = 0;
    task.colsA = 0;
    task.rowsB = 0;
    task.colsB = 0;
  }

  EXPECT_FALSE(task.ValidationImpl());
}

TEST(konkov_i_SparseMatmulTest_all, ComplexTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs_all::SparseMatmulTask task(task_data);

  if (world.rank() == 0) {
    task.A_values = {1.0, 2.0};
    task.A_row_indices = {0, 2};
    task.A_col_ptr = {0, 1, 1, 2};
    task.rowsA = 3;
    task.colsA = 3;

    task.B_values = {3.0, 4.0};
    task.B_row_indices = {1, 2};
    task.B_col_ptr = {0, 0, 1, 2};
    task.rowsB = 3;
    task.colsB = 3;
  }

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (world.rank() == 0) {
    std::vector<double> expected_values = {8.0};
    EXPECT_EQ(task.C_values, expected_values);
  }
}

TEST(konkov_i_SparseMatmulTest_all, IdentityMatrixTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs_all::SparseMatmulTask task(task_data);

  if (world.rank() == 0) {
    task.A_values = {1.0, 1.0, 1.0};
    task.A_row_indices = {0, 1, 2};
    task.A_col_ptr = {0, 1, 2, 3};
    task.rowsA = 3;
    task.colsA = 3;

    task.B_values = {1.0, 2.0, 3.0};
    task.B_row_indices = {0, 1, 2};
    task.B_col_ptr = {0, 1, 2, 3};
    task.rowsB = 3;
    task.colsB = 3;
  }

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (world.rank() == 0) {
    EXPECT_EQ(task.C_values, task.B_values);
    EXPECT_EQ(task.C_row_indices, task.B_row_indices);
    EXPECT_EQ(task.C_col_ptr, task.B_col_ptr);
  }
}