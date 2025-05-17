#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

void CreateTaskData(std::shared_ptr<ppc::core::TaskData> &task_data, int m, int k, int n, std::vector<double> &a_values,
                    std::vector<double> &a_row_indices, std::vector<double> &a_col_ptr, std::vector<double> &b_values,
                    std::vector<double> &b_row_indices, std::vector<double> &b_col_ptr, std::vector<double> &c_values,
                    std::vector<double> &c_row_indices, std::vector<double> &c_col_ptr) {
  task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(m);
  task_data->inputs_count.emplace_back(k);
  task_data->inputs_count.emplace_back(n);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_values.data()));
  task_data->inputs_count.emplace_back(a_values.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_indices.data()));
  task_data->inputs_count.emplace_back(a_row_indices.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data->inputs_count.emplace_back(a_col_ptr.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_values.data()));
  task_data->inputs_count.emplace_back(b_values.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_indices.data()));
  task_data->inputs_count.emplace_back(b_row_indices.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data->inputs_count.emplace_back(b_col_ptr.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_values.data()));
  task_data->outputs_count.emplace_back(c_values.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_indices.data()));
  task_data->outputs_count.emplace_back(c_row_indices.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data->outputs_count.emplace_back(c_col_ptr.size());
}

void CheckVectors(const std::vector<double> &expected, const std::vector<double> &actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_NEAR(expected[i], actual[i], 1e-9);
  }
}

void AssertResult(const std::vector<double> &c_values, const std::vector<double> &r_values,
                  const std::vector<double> &c_row_indices, const std::vector<double> &r_row_indices,
                  const std::vector<double> &c_col_ptr, const std::vector<double> &r_col_ptr) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    CheckVectors(c_values, r_values);
    CheckVectors(c_row_indices, r_row_indices);
    CheckVectors(c_col_ptr, r_col_ptr);
  }
}

void CreateAndRunTask(std::shared_ptr<ppc::core::TaskData> &task_data) {
  sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
}

}  // namespace

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_all, test_3x3_x_3x3) {
  int m = 3, k = 3, n = 3;

  std::vector<double> a_values = {2, 4, 1, 3, 5};
  std::vector<double> a_row_indices = {1, 2, 0, 1, 2};
  std::vector<double> a_col_ptr = {0, 2, 3, 5};

  std::vector<double> b_values = {1, 2, 3, 4};
  std::vector<double> b_row_indices = {0, 2, 0, 1};
  std::vector<double> b_col_ptr = {0, 2, 2, 4};

  std::vector<double> c_values(5);
  std::vector<double> c_row_indices(5);
  std::vector<double> c_col_ptr(4);

  std::shared_ptr<ppc::core::TaskData> task_data;
  CreateTaskData(task_data, m, k, n, a_values, a_row_indices, a_col_ptr, b_values, b_row_indices, b_col_ptr, c_values,
                 c_row_indices, c_col_ptr);

  CreateAndRunTask(task_data);

  std::vector<double> r_values = {8, 14, 4, 6, 12};
  std::vector<double> r_row_indices = {1, 2, 0, 1, 2};
  std::vector<double> r_col_ptr = {0, 2, 2, 5};

  AssertResult(c_values, r_values, c_row_indices, r_row_indices, c_col_ptr, r_col_ptr);
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_all, test_2x3_x_3x2) {
  int m = 2, k = 3, n = 2;

  std::vector<double> a_values = {1.0, 2.0, 3.0};
  std::vector<double> a_row_indices = {0, 1, 1};
  std::vector<double> a_col_ptr = {0, 1, 2, 3};

  std::vector<double> b_values = {1.0, 4.0, 5.0};
  std::vector<double> b_row_indices = {1, 0, 2};
  std::vector<double> b_col_ptr = {0, 1, 3};

  std::vector<double> c_values(3);
  std::vector<double> c_row_indices(3);
  std::vector<double> c_col_ptr(2);

  std::shared_ptr<ppc::core::TaskData> task_data;
  CreateTaskData(task_data, m, k, n, a_values, a_row_indices, a_col_ptr, b_values, b_row_indices, b_col_ptr, c_values,
                 c_row_indices, c_col_ptr);

  CreateAndRunTask(task_data);

  std::vector<double> r_values = {2.0, 4.0, 15.0};
  std::vector<double> r_row_indices = {1, 0, 1};
  std::vector<double> r_col_ptr = {0, 1};

  AssertResult(c_values, r_values, c_row_indices, r_row_indices, c_col_ptr, r_col_ptr);
}

TEST(sorokin_a_multiplication_sparse_matrices_double_ccs_all, test_3x2_x_2x4) {
  int m = 3, k = 2, n = 4;

  std::vector<double> a_values = {1.0, 3.0, 2.0};
  std::vector<double> a_row_indices = {1, 2, 0};
  std::vector<double> a_col_ptr = {0, 2, 3};

  std::vector<double> b_values = {4.0, 1.0, 5.0};
  std::vector<double> b_row_indices = {1, 0, 1};
  std::vector<double> b_col_ptr = {0, 1, 2, 2, 3};

  std::vector<double> c_values(4);
  std::vector<double> c_row_indices(4);
  std::vector<double> c_col_ptr(5);

  std::shared_ptr<ppc::core::TaskData> task_data;
  CreateTaskData(task_data, m, k, n, a_values, a_row_indices, a_col_ptr, b_values, b_row_indices, b_col_ptr, c_values,
                 c_row_indices, c_col_ptr);

  CreateAndRunTask(task_data);

  std::vector<double> r_values = {8.0, 1.0, 3.0, 10.0};
  std::vector<double> r_row_indices = {0, 1, 2, 0};
  std::vector<double> r_col_ptr = {0, 1, 3, 3, 4};

  AssertResult(c_values, r_values, c_row_indices, r_row_indices, c_col_ptr, r_col_ptr);
}

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
