#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sdobnov_v_complex_ccs_matrix_mult/include/complex_ccs_matrix_mult.hpp"

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_IncompatibleDimensions) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1(2, 3), m2(4, 5), result(2, 5);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_FALSE(task.ValidationImpl());
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_EmptyMatrices) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1(2, 3), m2(3, 4), result(2, 4);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  ASSERT_EQ(result.values.size(), 0);
  ASSERT_EQ(result.row_i.size(), 0);
  for (int p : result.col_p) ASSERT_EQ(p, 0);
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_1x1) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1(1, 1), m2(1, 1), result(1, 1);
  m1.addValue(0, 0, {0.0, 1.0});
  m2.addValue(0, 0, {0.0, -1.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);

  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS expected(1, 1);
  expected.addValue(0, 0, {1.0, 0.0});

  ASSERT_EQ(result, expected);
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_2x3_3x2) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS A(2, 3);
  A.addValue(0, 0, {1.0, 0.0});
  A.addValue(1, 1, {0.0, 1.0});
  A.addValue(2, 1, {1.0, -1.0});
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS B(3, 2);
  B.addValue(0, 0, {0.0, 1.0});
  B.addValue(1, 0, {2.0, 0.0});
  B.addValue(0, 1, {1.0, 0.0});
  B.addValue(0, 2, {1.0, 1.0});
  B.addValue(1, 2, {0.0, 2.0});
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS Result(2, 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&A), reinterpret_cast<uint8_t*>(&B)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&Result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS expected(2, 2);
  expected.addValue(0, 0, {0.0, 1.0});
  expected.addValue(1, 0, {2.0, 0.0});
  expected.addValue(0, 1, {2.0, 1.0});
  expected.addValue(1, 1, {2.0, 2.0});
  printf("C\n");
  ASSERT_EQ(Result, expected);
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_RandomSparseMatrices) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1 =
      sdobnov_v_complex_ccs_matrix_mult::generateRandomMatrix(10, 5, 0.1, 1);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m2 =
      sdobnov_v_complex_ccs_matrix_mult::generateRandomMatrix(5, 8, 0.1, 2);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS result(10, 8);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  ASSERT_EQ(result.col_p.size(), 9);
  ASSERT_EQ(result.values.size(), result.row_i.size());
}