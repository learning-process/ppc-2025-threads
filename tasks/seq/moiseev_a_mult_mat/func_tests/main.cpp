#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
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
  auto A = GenerateRandomMatrix(kSize, kSize);
  auto B = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> C(kSize * kSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(C.empty());
}

TEST(moiseev_a_mult_mat_seq, test_small_matrix) {
  auto A = GenerateRandomMatrix(2, 2);
  auto B = GenerateRandomMatrix(2, 2);
  std::vector<double> C(2 * 2, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(C.empty());
}

TEST(moiseev_a_mult_mat_seq, test_identity_matrix) {
  constexpr size_t kSize = 3;
  std::vector<double> A = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  auto B = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> C(kSize * kSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(C, B);
}

TEST(moiseev_a_mult_mat_seq, test_zero_matrix) {
  constexpr size_t kSize = 4;
  auto A = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> B(kSize * kSize, 0.0);
  std::vector<double> C(kSize * kSize, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  moiseev_a_mult_mat_seq::MultMatSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(C, B);
}
