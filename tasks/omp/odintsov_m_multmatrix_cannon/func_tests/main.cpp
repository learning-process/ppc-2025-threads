#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/odintsov_m_multmatrix_cannon/include/ops_omp.hpp"

std::vector<double> GenerateMatrix(int sz) {
  std::vector<double> matrix(sz * sz);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 10.0);
  for (int i = 0; i < sz; ++i) {
    for (int j = 0; j < sz; ++j) {
      matrix[i * sz + j] = dis(gen);
    }
  }
  return matrix;
}

std::vector<double> MultiplyMatrices(const std::vector<double> &A, const std::vector<double> &B, int n) {
  std::vector<double> C(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
  return C;
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_matrix_4) {
  std::vector<double> matrix_a = GenerateMatrix(4);
  std::vector<double> matrix_b = GenerateMatrix(4);
  std::vector<double> out_omp(16, 0);
  std::vector<double> out_ans = MultiplyMatrices(matrix_a, matrix_b, 4);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_matrix_100) {
  std::vector<double> matrix_a = GenerateMatrix(100);
  std::vector<double> matrix_b = GenerateMatrix(100);
  std::vector<double> out_omp(10000, 0);
  std::vector<double> out_ans = MultiplyMatrices(matrix_a, matrix_b, 100);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_matrix_900) {
  std::vector<double> matrix_a = GenerateMatrix(30);
  std::vector<double> matrix_b = GenerateMatrix(30);
  std::vector<double> out_omp(900, 0);
  std::vector<double> out_ans = MultiplyMatrices(matrix_a, matrix_b, 30);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_sz_block_1) {
  std::vector<double> matrix_a = GenerateMatrix(3);
  std::vector<double> matrix_b = GenerateMatrix(3);
  std::vector<double> out_omp(9, 0);
  std::vector<double> out_ans = MultiplyMatrices(matrix_a, matrix_b, 3);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  for (size_t i = 0; i < out_ans.size(); ++i) {
    EXPECT_NEAR(out_ans[i], out_omp[i], 1e-9);
  }
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_validation) {
  std::vector<double> matrix_a(12, 0);
  std::vector<double> matrix_b(12, 0);
  std::vector<double> out_omp(12, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());
  task_data_omp->inputs_count.emplace_back(matrix_b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));

  odintsov_m_mulmatrix_cannon_omp::MulMatrixCannonOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), false);
}
