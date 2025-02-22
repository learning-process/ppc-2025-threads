#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"
namespace {
void MatrixMultiplication(const std::vector<double> &matrixA, const std::vector<double> &matrixB,
                          std::vector<double> &resultMatrix, size_t matrixSize);
std::vector<double> getRandomMatrix(size_t size);

std::vector<double> getRandomMatrix(size_t size) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(1e-3, 1e3);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}

void MatrixMultiplication(const std::vector<double> &matrixA, const std::vector<double> &matrixB,
                          std::vector<double> &resultMatrix, size_t matrixSize) {
  for (size_t row = 0; row < matrixSize; ++row) {
    for (size_t col = 0; col < matrixSize; ++col) {
      resultMatrix[row * matrixSize + col] = 0.0;
      for (size_t k = 0; k < matrixSize; ++k) {
        resultMatrix[row * matrixSize + col] += matrixA[row * matrixSize + k] * matrixB[k * matrixSize + col];
      }
      resultMatrix[row * matrixSize + col] = round(resultMatrix[row * matrixSize + col] * 10000) / 10000;
    }
  }
}
}  // namespace
TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Identity) {
  size_t N = 3;
  size_t block_size = 2;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> C(N * N, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.Validation(), true);
  matrixMultiplication.PreProcessing();
  matrixMultiplication.Run();
  matrixMultiplication.PostProcessing();
  EXPECT_EQ(C, A);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Arbitrary_Values) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A = {2, 3, 1, 4, 0, 5, 1, 2, 3};
  std::vector<double> B = {1, 2, 3, 0, 1, 0, 4, 0, 1};
  std::vector<double> C(N * N, 0);
  std::vector<double> C_expected = {6.0, 7.0, 7.0, 24.0, 8.0, 17.0, 13.0, 4.0, 6.0};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), true);
  matrixMultiplication.PreProcessingImpl();
  matrixMultiplication.RunImpl();
  matrixMultiplication.PostProcessingImpl();
  EXPECT_EQ(C, C_expected);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_matrix_7x7) {
  size_t N = 7;
  size_t block_size = 3;
  std::vector<double> A = getRandomMatrix(N);
  std::vector<double> B = getRandomMatrix(N);
  std::vector<double> C(N * N, 0);
  std::vector<double> C_expected(N * N, 0);
  MatrixMultiplication(A, B, C_expected, N);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), true);
  matrixMultiplication.PreProcessingImpl();
  matrixMultiplication.RunImpl();
  matrixMultiplication.PostProcessingImpl();
  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(C_expected[i], C[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_matrix_16x16) {
  size_t N = 16;
  size_t block_size = 4;
  std::vector<double> A = getRandomMatrix(N);
  std::vector<double> B = getRandomMatrix(N);
  std::vector<double> C(N * N, 0);
  std::vector<double> C_expected(N * N, 0);
  MatrixMultiplication(A, B, C_expected, N);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), true);
  matrixMultiplication.PreProcessingImpl();
  matrixMultiplication.RunImpl();
  matrixMultiplication.PostProcessingImpl();
  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(C_expected[i], C[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_matrix_11x11) {
  size_t N = 11;
  size_t block_size = 10;
  std::vector<double> A = getRandomMatrix(N);
  std::vector<double> B = getRandomMatrix(N);
  std::vector<double> C(N * N, 0);
  std::vector<double> C_expected(N * N, 0);
  MatrixMultiplication(A, B, C_expected, N);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), true);
  matrixMultiplication.PreProcessingImpl();
  matrixMultiplication.RunImpl();
  matrixMultiplication.PostProcessingImpl();
  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(C_expected[i], C[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_matrix_1x1) {
  size_t N = 1;
  size_t block_size = 3;
  std::vector<double> A = {2};
  std::vector<double> B = {2};
  std::vector<double> C(N * N, 0);
  std::vector<double> C_expected(N * N, 4);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), true);
  matrixMultiplication.PreProcessingImpl();
  matrixMultiplication.RunImpl();
  matrixMultiplication.PostProcessingImpl();
  EXPECT_EQ(C, C_expected);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Empty_MatrixA) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A(0, 0);
  std::vector<double> B = {2, 98, 7, 6, 5, 4, 5, 6, 6};
  std::vector<double> C(N * N, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), false);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Empty_MatrixB) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A = {2, 98, 7, 6, 5, 4, 5, 6, 6};
  std::vector<double> B(0, 0);
  std::vector<double> C(N * N, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.ValidationImpl(), false);
}
