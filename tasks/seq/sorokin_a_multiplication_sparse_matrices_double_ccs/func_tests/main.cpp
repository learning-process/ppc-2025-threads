#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_seq.hpp"

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_1) {
  int M = 3;
  int K = 3;
  int N = 3;

  std::vector<double> A_values = {2, 4, 1, 3, 5};
  std::vector<double> A_row_indices = {1, 2, 0, 1, 2};
  std::vector<double> A_col_ptr = {0, 2, 3, 5};

  std::vector<double> B_values = {1, 2, 3, 4};
  std::vector<double> B_row_indices = {0, 2, 0, 1};
  std::vector<double> B_col_ptr = {0, 2, 2, 4};

  std::vector<double> C_values(5);
  std::vector<double> C_row_indices(5);
  std::vector<double> C_col_ptr(4);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  /*std::cout << std::endl;
  for (double value : C_values) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  for (double value : C_row_indices) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  for (double value : C_col_ptr) {
    std::cout << value << " ";
  }
  std::cout << std::endl;*/
  std::vector<double> R_values = {8, 14, 4, 6, 12};
  std::vector<double> R_row_indices = {1, 2, 0, 1, 2};
  std::vector<double> R_col_ptr = {0, 2, 2, 5};
  for (int i = 0; i < R_values.size(); i++) {
    ASSERT_NEAR(C_values[i], R_values[i], 1e-9);
  }
  for (int i = 0; i < R_row_indices.size(); i++) {
    ASSERT_NEAR(C_row_indices[i], R_row_indices[i], 1e-9);
  }
  for (int i = 0; i < R_col_ptr.size(); i++) {
    ASSERT_NEAR(C_col_ptr[i], R_col_ptr[i], 1e-9);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_2) {
  int M = 2;
  int K = 3;
  int N = 2;

  std::vector<double> A_values = {1.0, 2.0, 3.0};
  std::vector<double> A_row_indices = {0, 1, 1};
  std::vector<double> A_col_ptr = {0, 1, 2, 3};

  std::vector<double> B_values = {1.0, 4.0, 5.0};
  std::vector<double> B_row_indices = {1, 0, 2};
  std::vector<double> B_col_ptr = {0, 1, 3};

  std::vector<double> C_values(5);
  std::vector<double> C_row_indices(5);
  std::vector<double> C_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> R_values = {2.0, 4.0, 15.0};
  std::vector<double> R_row_indices = {1, 0, 1};
  std::vector<double> R_col_ptr = {0, 1};
  for (int i = 0; i < R_values.size(); i++) {
    ASSERT_NEAR(C_values[i], R_values[i], 1e-9);
  }
  for (int i = 0; i < R_row_indices.size(); i++) {
    ASSERT_NEAR(C_row_indices[i], R_row_indices[i], 1e-9);
  }
  for (int i = 0; i < R_col_ptr.size(); i++) {
    ASSERT_NEAR(C_col_ptr[i], R_col_ptr[i], 1e-9);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_3) {
  int M = 3;
  int K = 2;
  int N = 4;

  std::vector<double> A_values = {1.0, 3.0, 2.0};
  std::vector<double> A_row_indices = {1, 2, 0};
  std::vector<double> A_col_ptr = {0, 2, 3};

  std::vector<double> B_values = {4.0, 1.0, 5.0};
  std::vector<double> B_row_indices = {1, 0, 1};
  std::vector<double> B_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> C_values(5);
  std::vector<double> C_row_indices(5);
  std::vector<double> C_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> R_values = {8.0, 1.0, 3.0, 10.0};
  std::vector<double> R_row_indices = {0, 1, 2, 0};
  std::vector<double> R_col_ptr = {0, 1, 3, 3, 4};
  for (int i = 0; i < R_values.size(); i++) {
    ASSERT_NEAR(C_values[i], R_values[i], 1e-9);
  }
  for (int i = 0; i < R_row_indices.size(); i++) {
    ASSERT_NEAR(C_row_indices[i], R_row_indices[i], 1e-9);
  }
  for (int i = 0; i < R_col_ptr.size(); i++) {
    ASSERT_NEAR(C_col_ptr[i], R_col_ptr[i], 1e-9);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_val_1) {
  int M = 3;
  int K = 0;
  int N = 4;

  std::vector<double> A_values = {1.0, 3.0, 2.0};
  std::vector<double> A_row_indices = {1, 2, 0};
  std::vector<double> A_col_ptr = {0, 2, 3};

  std::vector<double> B_values = {4.0, 1.0, 5.0};
  std::vector<double> B_row_indices = {1, 0, 1};
  std::vector<double> B_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> C_values(5);
  std::vector<double> C_row_indices(5);
  std::vector<double> C_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_val_2) {
  int M = 0;
  int K = 2;
  int N = 4;

  std::vector<double> A_values = {1.0, 3.0, 2.0};
  std::vector<double> A_row_indices = {1, 2, 0};
  std::vector<double> A_col_ptr = {0, 2, 3};

  std::vector<double> B_values = {4.0, 1.0, 5.0};
  std::vector<double> B_row_indices = {1, 0, 1};
  std::vector<double> B_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> C_values(5);
  std::vector<double> C_row_indices(5);
  std::vector<double> C_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(M);
  task_data_seq->inputs_count.emplace_back(K);
  task_data_seq->inputs_count.emplace_back(N);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_values.data()));
  task_data_seq->inputs_count.emplace_back(A_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(A_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(A_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_values.data()));
  task_data_seq->inputs_count.emplace_back(B_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(B_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(B_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_values.data()));
  task_data_seq->outputs_count.emplace_back(C_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(C_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(C_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
