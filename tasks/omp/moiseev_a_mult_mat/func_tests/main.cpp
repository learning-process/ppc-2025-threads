#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/moiseev_a_mult_mat/include/ops_omp.hpp"

namespace {

std::vector<double> GenerateRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (auto &val : matrix) {
    val = dist(gen);
  }
  return matrix;
}

}  // namespace

TEST(moiseev_a_mult_mat_omp, test_large_matrix) {
  constexpr size_t kSize = 100;
  auto a = GenerateRandomMatrix(kSize, kSize);
  auto b = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_omp, test_small_matrix) {
  auto a = GenerateRandomMatrix(2, 2);
  auto b = GenerateRandomMatrix(2, 2);
  std::vector<double> c(2 * 2, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_omp, test_identity_matrix) {
  constexpr size_t kSize = 3;
  std::vector<double> a = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  auto b = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  for (size_t i = 0; i < c.size(); ++i) {
    EXPECT_DOUBLE_EQ(c[i], b[i]);
  }
}

TEST(moiseev_a_mult_mat_omp, test_zero_matrix) {
  constexpr size_t kSize = 4;
  auto a = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> b(kSize * kSize, 0.0);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(c, b);
}
