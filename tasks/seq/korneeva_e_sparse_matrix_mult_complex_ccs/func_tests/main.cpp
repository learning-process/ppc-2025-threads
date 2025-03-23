#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_seq.hpp"

namespace korneeva_e_ccs = korneeva_e_sparse_matrix_mult_complex_ccs_seq;

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_incompatible_sizes) {
  korneeva_e_ccs::SparseMatrixCCS m1(2, 3, 0);
  m1.col_offsets = {0, 0, 0, 0};
  korneeva_e_ccs::SparseMatrixCCS m2(2, 2, 0);
  m2.col_offsets = {0, 0, 0};
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_FALSE(korneeva_e_ccs::RunTask(m1, m2, result));
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_negative_dimensions) {
  korneeva_e_ccs::SparseMatrixCCS m1(-1, 2, 0);
  m1.col_offsets = {0, 0, 0};
  korneeva_e_ccs::SparseMatrixCCS m2(2, 2, 0);
  m2.col_offsets = {0, 0, 0};
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_FALSE(korneeva_e_ccs::RunTask(m1, m2, result));
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_empty_input) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  korneeva_e_ccs::SparseMatrixCCS result;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  korneeva_e_ccs::SparseMatrixMultComplexCCS task(task_data);
  ASSERT_FALSE(task.ValidationImpl());
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_identity_mult) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_left_identity_mult) {
  auto i = korneeva_e_ccs::CreateCcsFromDense(
      {{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
       {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
       {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0)}});

  auto a =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 2.0), korneeva_e_ccs::Complex(3.0, 4.0)},
                                          {korneeva_e_ccs::Complex(5.0, 0.0), korneeva_e_ccs::Complex(7.0, 8.0)},
                                          {korneeva_e_ccs::Complex(9.0, 10.0), korneeva_e_ccs::Complex(11.0, 12.0)}});

  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(i, a, result));

  korneeva_e_ccs::ExpectMatrixEq(result, a);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_zero_matrix) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_full_zero_matrix) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_diagonal_matrices) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(2.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(3.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(4.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(5.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(8.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(15.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_complex_numbers) {
  auto m1 = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 1.0)}});
  auto m2 = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, -1.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_large_complex_values) {
  auto m1 = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1e10, 1e10)}});
  auto m2 = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1e10, -1e10)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(2e20, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_rectangular_matrices) {
  auto m1 = korneeva_e_ccs::CreateCcsFromDense(
      {{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0), korneeva_e_ccs::Complex(3.0, 0.0)},
       {korneeva_e_ccs::Complex(4.0, 0.0), korneeva_e_ccs::Complex(5.0, 0.0), korneeva_e_ccs::Complex(6.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(7.0, 0.0), korneeva_e_ccs::Complex(8.0, 0.0)},
                                          {korneeva_e_ccs::Complex(9.0, 0.0), korneeva_e_ccs::Complex(10.0, 0.0)},
                                          {korneeva_e_ccs::Complex(11.0, 0.0), korneeva_e_ccs::Complex(12.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(58.0, 0.0), korneeva_e_ccs::Complex(64.0, 0.0)},
                                          {korneeva_e_ccs::Complex(139.0, 0.0), korneeva_e_ccs::Complex(154.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_matrix_vector_mult) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 2.0), korneeva_e_ccs::Complex(3.0, 4.0)},
                                          {korneeva_e_ccs::Complex(5.0, 6.0), korneeva_e_ccs::Complex(7.0, 8.0)}});
  auto vec =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0)}, {korneeva_e_ccs::Complex(2.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, vec, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(7.0, 10.0)}, {korneeva_e_ccs::Complex(19.0, 22.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_vector_matrix_mult) {
  auto vec =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(3.0, 0.0), korneeva_e_ccs::Complex(4.0, 0.0)},
                                          {korneeva_e_ccs::Complex(5.0, 0.0), korneeva_e_ccs::Complex(6.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(vec, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(13.0, 0.0), korneeva_e_ccs::Complex(16.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_matrix_unit_vector) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 1.0), korneeva_e_ccs::Complex(2.0, 2.0)},
                                          {korneeva_e_ccs::Complex(3.0, 3.0), korneeva_e_ccs::Complex(4.0, 4.0)}});
  auto vec =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0)}, {korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, vec, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 1.0)}, {korneeva_e_ccs::Complex(3.0, 3.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_large_size_matrices) {
  std::mt19937 gen(789);
  auto m1 = korneeva_e_ccs::CreateRandomMatrix(1000, 1000, 10000, gen);
  auto m2 = korneeva_e_ccs::CreateRandomMatrix(1000, 1000, 10000, gen);
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  ASSERT_EQ(result.rows, 1000);
  ASSERT_EQ(result.cols, 1000);
  EXPECT_LE(result.nnz, 1000000);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_sparse_matrices) {
  auto m1 = korneeva_e_ccs::CreateCcsFromDense(
      {{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
       {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(3.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(4.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(4.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_mixed_values) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(3.0, 0.0)},
                                          {korneeva_e_ccs::Complex(4.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(3.0, 0.0)},
                                          {korneeva_e_ccs::Complex(8.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_dense_sparse_mult) {
  auto m1 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0)},
                                          {korneeva_e_ccs::Complex(3.0, 0.0), korneeva_e_ccs::Complex(4.0, 0.0)}});
  auto m2 =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  auto expected =
      korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                          {korneeva_e_ccs::Complex(3.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)}});
  korneeva_e_ccs::ExpectMatrixEq(result, expected);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_random_matrices1) {
  std::mt19937 gen(42);
  auto m1 = korneeva_e_ccs::CreateRandomMatrix(2, 2, 2, gen);
  auto m2 = korneeva_e_ccs::CreateRandomMatrix(2, 2, 2, gen);
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  ASSERT_EQ(result.rows, 2);
  ASSERT_EQ(result.cols, 2);
  EXPECT_LE(result.nnz, 4);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_random_matrices2) {
  std::mt19937 gen(123);
  auto m1 = korneeva_e_ccs::CreateRandomMatrix(100, 100, 500, gen);
  auto m2 = korneeva_e_ccs::CreateRandomMatrix(100, 100, 500, gen);
  korneeva_e_ccs::SparseMatrixCCS result;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(m1, m2, result));

  ASSERT_EQ(result.rows, 100);
  ASSERT_EQ(result.cols, 100);
  EXPECT_LE(result.nnz, 10000);
}

TEST(korneeva_e_sparse_matrix_mult_complex_ccs_seq, test_associativity) {
  auto a = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0)},
                                               {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(3.0, 0.0)}});
  auto b = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(4.0, 0.0), korneeva_e_ccs::Complex(0.0, 0.0)},
                                               {korneeva_e_ccs::Complex(5.0, 0.0), korneeva_e_ccs::Complex(6.0, 0.0)}});
  auto c = korneeva_e_ccs::CreateCcsFromDense({{korneeva_e_ccs::Complex(1.0, 0.0), korneeva_e_ccs::Complex(2.0, 0.0)},
                                               {korneeva_e_ccs::Complex(0.0, 0.0), korneeva_e_ccs::Complex(1.0, 0.0)}});

  korneeva_e_ccs::SparseMatrixCCS ab;
  korneeva_e_ccs::SparseMatrixCCS ab_c;
  korneeva_e_ccs::SparseMatrixCCS bc;
  korneeva_e_ccs::SparseMatrixCCS a_bc;

  ASSERT_TRUE(korneeva_e_ccs::RunTask(a, b, ab));
  ASSERT_TRUE(korneeva_e_ccs::RunTask(ab, c, ab_c));

  ASSERT_TRUE(korneeva_e_ccs::RunTask(b, c, bc));
  ASSERT_TRUE(korneeva_e_ccs::RunTask(a, bc, a_bc));

  korneeva_e_ccs::ExpectMatrixEq(ab_c, a_bc);
}