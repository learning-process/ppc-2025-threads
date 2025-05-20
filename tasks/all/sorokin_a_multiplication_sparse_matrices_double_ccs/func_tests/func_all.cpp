#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_all, test_val_k_0) {
  boost::mpi::communicator world;
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
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(k);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_tbb->inputs_count.emplace_back(a_values.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_tbb->inputs_count.emplace_back(a_row_indices.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_tbb->inputs_count.emplace_back(a_col_ptr.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_tbb->inputs_count.emplace_back(b_values.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_tbb->inputs_count.emplace_back(b_row_indices.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_tbb->inputs_count.emplace_back(b_col_ptr.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_tbb->outputs_count.emplace_back(c_values.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_tbb->outputs_count.emplace_back(c_row_indices.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_tbb->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL test_task_tbb(task_data_tbb);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_tbb.Validation(), false);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_all, test_val_m_0) {
  boost::mpi::communicator world;
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
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(k);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_tbb->inputs_count.emplace_back(a_values.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_tbb->inputs_count.emplace_back(a_row_indices.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_tbb->inputs_count.emplace_back(a_col_ptr.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_tbb->inputs_count.emplace_back(b_values.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_tbb->inputs_count.emplace_back(b_row_indices.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_tbb->inputs_count.emplace_back(b_col_ptr.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_tbb->outputs_count.emplace_back(c_values.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_tbb->outputs_count.emplace_back(c_row_indices.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_tbb->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL test_task_tbb(task_data_tbb);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_tbb.Validation(), false);
  }
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_all, test_val_n_0) {
  boost::mpi::communicator world;
  int m = 4;
  int k = 2;
  int n = 0;

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
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs_count.emplace_back(m);
  task_data_tbb->inputs_count.emplace_back(k);
  task_data_tbb->inputs_count.emplace_back(n);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data_tbb->inputs_count.emplace_back(a_values.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data_tbb->inputs_count.emplace_back(a_row_indices.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data_tbb->inputs_count.emplace_back(a_col_ptr.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data_tbb->inputs_count.emplace_back(b_values.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data_tbb->inputs_count.emplace_back(b_row_indices.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data_tbb->inputs_count.emplace_back(b_col_ptr.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data_tbb->outputs_count.emplace_back(c_values.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data_tbb->outputs_count.emplace_back(c_row_indices.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data_tbb->outputs_count.emplace_back(c_col_ptr.size());

  // Create Task
  sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL test_task_tbb(task_data_tbb);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_tbb.Validation(), false);
  }
}
