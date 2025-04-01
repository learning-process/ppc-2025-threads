#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

namespace yasakova_t_sparse_matrix_multiplication_omp {
MatrixStructure static RandMatrix(uint32_t num_rows, uint32_t num_cols, double non_zero_percentage) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> distr(-10000, 10000);
  MatrixStructure result{
      .num_rows = num_rows, .num_cols = num_cols, .task_data = std::vector<std::complex<double>>(num_rows * num_cols)};
  std::ranges::generate(result.task_data, [&]() {
    const auto value = distr(gen);
    const auto real_part = (value < (distr.min() + ((distr.max() - distr.min()) * non_zero_percentage))) ? value : 0;

    std::complex<double> complex_num;
    complex_num.real(real_part);
    if (real_part != 0.0) {
      complex_num.imag(distr(gen));
    }

    return complex_num;
  });
  return result;
}
void static TestCRSMatrixMultiplication(MatrixStructure &&matrix_left, MatrixStructure &&matrix_right) {
  SparseMatrixFormat crs_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat crs_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat crs_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&crs_left), reinterpret_cast<uint8_t *>(&crs_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&crs_result)};
  task_data->outputs_count = {1};

  yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier task(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  MatrixStructure actual_result = ConvertFromCRS(crs_result);
  EXPECT_EQ(actual_result, MatrixMultiply(matrix_left, matrix_right));
}
}  // namespace yasakova_t_sparse_matrix_multiplication_omp

TEST(yasakova_t_sparse_matrix_multiplication_omp, MultiplySquareMatrices) {
  MatrixStructure matrix_left{
      .num_rows = 5, .num_cols = 5, .task_data = {43, 46, 21, 21, 87, 39, 26, 82, 65, 62, 97, 47, 32,
                                                  16, 61, 76, 43, 78, 50, 63, 18, 14, 84, 22, 55}};
  MatrixStructure matrix_right{
      .num_rows = 5, .num_cols = 5, .task_data = {43, 46, 21, 21, 87, 39, 26, 82, 65, 62, 97, 47, 32,
                                                  16, 61, 76, 43, 78, 50, 63, 18, 14, 84, 22, 55}};
  MatrixStructure ref{
      .num_rows = 5, .num_cols = 5, .task_data = {8842,  6282,  14293, 7193,  13982, 16701, 9987,  15853, 8435,
                                                  17512, 11422, 8730,  13287, 7746,  17668, 17445, 11312, 16810,
                                                  9525,  20651, 12130, 6856,  10550, 4942,  11969}};
  EXPECT_EQ(MatrixMultiply(matrix_left, matrix_right), ref);
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, MultiplyRectangularMatrices) {
  MatrixStructure matrix_left{.num_rows = 5, .num_cols = 4, .task_data = {43, 46, 21, 21, 39, 26, 82, 65, 97, 47,
                                                                          32, 16, 76, 43, 78, 50, 18, 14, 84, 22}};
  MatrixStructure matrix_right{.num_rows = 4, .num_cols = 5, .task_data = {43, 46, 21, 21, 87, 39, 26, 82, 65, 62,
                                                                           97, 47, 32, 16, 61, 76, 43, 78, 50, 63}};
  MatrixStructure ref{
      .num_rows = 5, .num_cols = 5, .task_data = {7276,  5064,  6985,  5279, 9197, 15585, 9119,  10645, 7071,
                                                  14102, 10324, 7876,  8163, 6404, 14313, 16311, 10430, 11518,
                                                  8139,  17186, 11140, 6086, 5930, 3732,  8944}};
  EXPECT_EQ(MatrixMultiply(matrix_left, matrix_right), ref);
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x30_FullyDense) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 30, .0),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 30, .0));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x30_20PercentNonZero) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 30, .20),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 30, .20));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x30_MixedDensity) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 30, .20),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 30, .50));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x40_VaryingDensity) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 40, .70),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(40, 30, .60));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrices30x23_70PercentNonZero) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 23, .70),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(23, 30, .63));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrix30x1_VeryHighDensity) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 1, .70),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(1, 30, .63));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, SparseMatrix30x1_LowDensity) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 1, .38),
      yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(1, 30, .63));
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, InverseMatrixMultiplication) {
  MatrixStructure matrix_left{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, 1, 0, 1}};
  MatrixStructure matrix_right{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, -1, 0, 1}};
  MatrixStructure ref{.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
  EXPECT_EQ(MatrixMultiply(matrix_left, matrix_right), ref);
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, test_crs_random_inv) {
  yasakova_t_sparse_matrix_multiplication_omp::TestCRSMatrixMultiplication(
      {.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, 1, 0, 1}},
      {.num_rows = 3, .num_cols = 3, .task_data = {1, 0, 0, 1, -1, 0, -1, 0, 1}});
}
TEST(yasakova_t_sparse_matrix_multiplication_omp, ValidationFailure_IncompatibleDimensions) {
  const auto matrix_left = yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(30, 40, .70);
  const auto matrix_right = yasakova_t_sparse_matrix_multiplication_omp::RandMatrix(50, 50, .70);

  SparseMatrixFormat crs_left = ConvertToCRS(matrix_left);
  SparseMatrixFormat crs_right = ConvertToCRS(matrix_right);
  SparseMatrixFormat crs_result;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&crs_left), reinterpret_cast<uint8_t *>(&crs_right)};
  task_data->inputs_count = {matrix_left.num_rows, matrix_left.num_cols, matrix_right.num_rows, matrix_right.num_cols};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&crs_result)};
  task_data->outputs_count = {1};

  yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier task(task_data);
  EXPECT_FALSE(task.Validation());
}
