#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/moiseev_a_mult_mat/include/ops_seq.hpp"

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

TEST(moiseev_a_mult_mat_seq, test_large_matrix) {
  constexpr size_t kSize = 100;
  auto a = GenerateRandomMatrix(kSize, kSize);
  auto b = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_seq, test_small_matrix) {
  auto a = GenerateRandomMatrix(2, 2);
  auto b = GenerateRandomMatrix(2, 2);
  std::vector<double> c(2 * 2, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_seq, test_matrix) {
  constexpr size_t kSize = 5;
  auto a = GenerateRandomMatrix(kSize, kSize);
  auto b = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_seq, test_zero_matrix) {
  constexpr size_t kSize = 4;
  auto a = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> b(kSize * kSize, 0.0);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(c, b);
}
