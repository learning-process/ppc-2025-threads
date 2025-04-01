#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

namespace {
MatrixStructure RandMatrix(uint32_t num_rows, uint32_t num_cols, double percentage) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> distr(-10000, 10000);
  MatrixStructure result{.num_rows = num_rows, .num_cols = num_cols, .elements = std::vector<std::complex<double>>(num_rows * num_cols)};
  std::ranges::generate(result.elements, [&]() {
    const auto el = distr(gen);
    const auto re = (el < (distr.min() + ((distr.max() - distr.min()) * percentage))) ? el : 0;

    std::complex<double> cmplx;
    cmplx.real(re);
    if (re != 0.0) {
      cmplx.imag(distr(gen));
    }

    return cmplx;
  });
  return result;
}
void TestMatrixCRS(MatrixStructure &&left_matrix, MatrixStructure &&right_matrix) {
  SparseMatrixCRS crs_lhs = ConvertToCRS(left_matrix);
  SparseMatrixCRS crs_rhs = ConvertToCRS(right_matrix);
  SparseMatrixCRS crs_out;

  auto elements = std::make_shared<ppc::core::TaskData>();
  elements->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  elements->inputs_count = {left_matrix.num_rows, left_matrix.num_cols, right_matrix.num_rows, right_matrix.num_cols};
  elements->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  elements->outputs_count = {1};

  yasakova_t_sparse_matrix_multiplication_omp::MatrixMultiplicationTask task(elements);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  MatrixStructure regular_out = ConvertFromCRS(crs_out);
  EXPECT_EQ(regular_out, MultiplyMatrices(left_matrix, right_matrix));
}
}  // namespace

// clang-format off
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_regular_matrix_mult_1) {
  MatrixStructure left_matrix{ .num_rows=5, .num_cols=5, .elements={
    43, 46, 21, 21, 87,
    39, 26, 82, 65, 62,
    97, 47, 32, 16, 61,
    76, 43, 78, 50, 63,
    18, 14, 84, 22, 55
  }};
  MatrixStructure right_matrix{ .num_rows=5, .num_cols=5, .elements={
    43, 46, 21, 21, 87,
    39, 26, 82, 65, 62,
    97, 47, 32, 16, 61,
    76, 43, 78, 50, 63,
    18, 14, 84, 22, 55
  }};
  MatrixStructure ref{ .num_rows=5, .num_cols=5, .elements={
    8842, 6282, 14293, 7193, 13982,
    16701, 9987, 15853, 8435, 17512,
    11422, 8730, 13287, 7746, 17668,
    17445, 11312, 16810, 9525, 20651,
    12130, 6856, 10550, 4942, 11969
  }};
  EXPECT_EQ(MultiplyMatrices(left_matrix, right_matrix), ref);
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_regular_matrix_mult_2) {
  MatrixStructure left_matrix{ .num_rows=5, .num_cols=4, .elements={
    43, 46, 21, 21,
    39, 26, 82, 65,
    97, 47, 32, 16,
    76, 43, 78, 50,
    18, 14, 84, 22
  }};
  MatrixStructure right_matrix{ .num_rows=4, .num_cols=5, .elements={
    43, 46, 21, 21, 87,
    39, 26, 82, 65, 62,
    97, 47, 32, 16, 61,
    76, 43, 78, 50, 63
  }};
  MatrixStructure ref{ .num_rows=5, .num_cols=5, .elements={
    7276, 5064, 6985, 5279, 9197,
    15585, 9119, 10645, 7071, 14102,
    10324, 7876, 8163, 6404, 14313,
    16311, 10430, 11518, 8139, 17186,
    11140, 6086, 5930, 3732, 8944
  }};
  EXPECT_EQ(MultiplyMatrices(left_matrix, right_matrix), ref);
}
// clang-format on

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x30p00mul30x30p00) {
  TestMatrixCRS(RandMatrix(30, 30, .0), RandMatrix(30, 30, .0));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x30p20mul30x30p20) {
  TestMatrixCRS(RandMatrix(30, 30, .20), RandMatrix(30, 30, .20));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x30p20mul30x30p50) {
  TestMatrixCRS(RandMatrix(30, 30, .20), RandMatrix(30, 30, .50));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x30p70mul30x30p50) {
  TestMatrixCRS(RandMatrix(30, 30, .70), RandMatrix(30, 30, .50));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x30p70mul30x30p20) {
  TestMatrixCRS(RandMatrix(30, 30, .70), RandMatrix(30, 30, .20));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x40p70mul40x30p60) {
  TestMatrixCRS(RandMatrix(30, 40, .70), RandMatrix(40, 30, .60));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x23p70mul23x30p63) {
  TestMatrixCRS(RandMatrix(30, 23, .70), RandMatrix(23, 30, .63));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x1p70mul1x1p63) {
  TestMatrixCRS(RandMatrix(30, 1, .70), RandMatrix(1, 30, .63));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_30x1p38mul1x1p63) {
  TestMatrixCRS(RandMatrix(30, 1, .38), RandMatrix(1, 30, .63));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_regular_matrix_mult_inv) {
  MatrixStructure left_matrix{.num_rows = 3, .num_cols = 3, .elements = {1, 0, 0, 1, -1, 0, 1, 0, 1}};
  MatrixStructure right_matrix{.num_rows = 3, .num_cols = 3, .elements = {1, 0, 0, 1, -1, 0, -1, 0, 1}};
  MatrixStructure ref{.num_rows = 3, .num_cols = 3, .elements = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
  EXPECT_EQ(MultiplyMatrices(left_matrix, right_matrix), ref);
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_inv) {
  TestMatrixCRS({.num_rows = 3, .num_cols = 3, .elements = {1, 0, 0, 1, -1, 0, 1, 0, 1}},
                {.num_rows = 3, .num_cols = 3, .elements = {1, 0, 0, 1, -1, 0, -1, 0, 1}});
}
TEST(tyrin_m_matmul_crs_complex_omp, validation_failure) {
  const auto left_matrix = RandMatrix(30, 40, .70);
  const auto right_matrix = RandMatrix(50, 50, .70);

  SparseMatrixCRS crs_lhs = ConvertToCRS(left_matrix);
  SparseMatrixCRS crs_rhs = ConvertToCRS(right_matrix);
  SparseMatrixCRS crs_out;

  auto elements = std::make_shared<ppc::core::TaskData>();
  elements->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  elements->inputs_count = {left_matrix.num_rows, left_matrix.num_cols, right_matrix.num_rows, right_matrix.num_cols};
  elements->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  elements->outputs_count = {1};

  yasakova_t_sparse_matrix_multiplication_omp::MatrixMultiplicationTask task(elements);
  EXPECT_FALSE(task.Validation());
}