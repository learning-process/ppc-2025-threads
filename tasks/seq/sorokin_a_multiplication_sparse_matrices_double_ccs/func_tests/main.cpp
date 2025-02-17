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
  int m = 3;
  int k = 3;
  int n = 3;

  std::vector<double> a_values = {2, 4, 1, 3, 5};
  std::vector<double> a_row_indices = {1, 2, 0, 1, 2};
  std::vector<double> a_col_ptr = {0, 2, 3, 5};

  std::vector<double> b_values = {1, 2, 3, 4};
  std::vector<double> b_row_indices = {0, 2, 0, 1};
  std::vector<double> b_col_ptr = {0, 2, 2, 4};

  std::vector<double> c_values(5);
  std::vector<double> c_row_indices(5);
  std::vector<double> c_col_ptr(4);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(k);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_seq->inputs_count.emplace_back(a_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(a_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(a_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_seq->inputs_count.emplace_back(b_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(b_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(b_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_seq->outputs_count.emplace_back(c_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(c_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> R_values = {8, 14, 4, 6, 12};
  std::vector<double> R_row_indices = {1, 2, 0, 1, 2};
  std::vector<double> R_col_ptr = {0, 2, 2, 5};
  for (size_t i = 0; i < R_values.size(); i++) {
    ASSERT_NEAR(c_values[i], R_values[i], 1e-9);
  }
  for (size_t i = 0; i < R_row_indices.size(); i++) {
    ASSERT_NEAR(c_row_indices[i], R_row_indices[i], 1e-9);
  }
  for (size_t i = 0; i < R_col_ptr.size(); i++) {
    ASSERT_NEAR(c_col_ptr[i], R_col_ptr[i], 1e-9);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_2) {
  int m = 2;
  int k = 3;
  int n = 2;

  std::vector<double> a_values = {1.0, 2.0, 3.0};
  std::vector<double> a_row_indices = {0, 1, 1};
  std::vector<double> a_col_ptr = {0, 1, 2, 3};

  std::vector<double> b_values = {1.0, 4.0, 5.0};
  std::vector<double> b_row_indices = {1, 0, 2};
  std::vector<double> b_col_ptr = {0, 1, 3};

  std::vector<double> c_values(5);
  std::vector<double> c_row_indices(5);
  std::vector<double> c_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(k);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_seq->inputs_count.emplace_back(a_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(a_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(a_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_seq->inputs_count.emplace_back(b_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(b_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(b_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_seq->outputs_count.emplace_back(c_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(c_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> R_values = {2.0, 4.0, 15.0};
  std::vector<double> R_row_indices = {1, 0, 1};
  std::vector<double> R_col_ptr = {0, 1};
  for (size_t i = 0; i < R_values.size(); i++) {
    ASSERT_NEAR(c_values[i], R_values[i], 1e-9);
  }
  for (size_t i = 0; i < R_row_indices.size(); i++) {
    ASSERT_NEAR(c_row_indices[i], R_row_indices[i], 1e-9);
  }
  for (size_t i = 0; i < R_col_ptr.size(); i++) {
    ASSERT_NEAR(c_col_ptr[i], R_col_ptr[i], 1e-9);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_3) {
  int m = 3;
  int k = 2;
  int n = 4;

  std::vector<double> a_values = {1.0, 3.0, 2.0};
  std::vector<double> a_row_indices = {1, 2, 0};
  std::vector<double> a_col_ptr = {0, 2, 3};

  std::vector<double> b_values = {4.0, 1.0, 5.0};
  std::vector<double> b_row_indices = {1, 0, 1};
  std::vector<double> b_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> c_values(5);
  std::vector<double> c_row_indices(5);
  std::vector<double> c_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(k);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_seq->inputs_count.emplace_back(a_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(a_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(a_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_seq->inputs_count.emplace_back(b_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(b_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(b_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_seq->outputs_count.emplace_back(c_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(c_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> R_values = {8.0, 1.0, 3.0, 10.0};
  std::vector<double> R_row_indices = {0, 1, 2, 0};
  std::vector<double> R_col_ptr = {0, 1, 3, 3, 4};
  for (size_t i = 0; i < R_values.size(); i++) {
    ASSERT_NEAR(c_values[i], R_values[i], 1e-9);
  }
  for (size_t i = 0; i < R_row_indices.size(); i++) {
    ASSERT_NEAR(c_row_indices[i], R_row_indices[i], 1e-9);
  }
  for (size_t i = 0; i < R_col_ptr.size(); i++) {
    ASSERT_NEAR(c_col_ptr[i], R_col_ptr[i], 1e-9);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_val_1) {
  int m = 3;
  int k = 0;
  int n = 4;

  std::vector<double> a_values = {1.0, 3.0, 2.0};
  std::vector<double> a_row_indices = {1, 2, 0};
  std::vector<double> a_col_ptr = {0, 2, 3};

  std::vector<double> b_values = {4.0, 1.0, 5.0};
  std::vector<double> b_row_indices = {1, 0, 1};
  std::vector<double> b_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> c_values(5);
  std::vector<double> c_row_indices(5);
  std::vector<double> c_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(k);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_seq->inputs_count.emplace_back(a_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(a_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(a_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_seq->inputs_count.emplace_back(b_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(b_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(b_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_seq->outputs_count.emplace_back(c_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(c_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_seq, test_val_2) {
  int m = 0;
  int k = 2;
  int n = 4;

  std::vector<double> a_values = {1.0, 3.0, 2.0};
  std::vector<double> a_row_indices = {1, 2, 0};
  std::vector<double> a_col_ptr = {0, 2, 3};

  std::vector<double> b_values = {4.0, 1.0, 5.0};
  std::vector<double> b_row_indices = {1, 0, 1};
  std::vector<double> b_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> c_values(5);
  std::vector<double> c_row_indices(5);
  std::vector<double> c_col_ptr(5);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(k);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_seq->inputs_count.emplace_back(a_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(a_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(a_col_ptr.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_seq->inputs_count.emplace_back(b_values.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_seq->inputs_count.emplace_back(b_row_indices.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_seq->inputs_count.emplace_back(b_col_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_seq->outputs_count.emplace_back(c_values.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_seq->outputs_count.emplace_back(c_row_indices.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_seq->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
