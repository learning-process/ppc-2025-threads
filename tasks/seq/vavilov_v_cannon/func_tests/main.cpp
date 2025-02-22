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

#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

std::vector<double> GenerateRandomMatrix(size_t N, double min_val = -10.0, double max_val = 10.0) {
  std::vector<double> matrix(N * N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (size_t i = 0; i < N * N; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

std::vector<double> MultiplyMatrices(const std::vector<double>& A, const std::vector<double>& B, size_t N) {
  std::vector<double> C(N * N, 0.0);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
  return C;
}

TEST(vavilov_v_cannon_seq, test_fixed_4x4) {
  constexpr size_t N = 4;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> B = {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
  std::vector<double> expected_output = {4, 6, 6, 4, 12, 14, 14, 12, 20, 22, 22, 20, 28, 30, 30, 28};
  std::vector<double> C(N * N, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(expected_output[i], C[i], 1e-6);
  }
}

TEST(vavilov_v_cannon_seq, test_random_500) {
  constexpr size_t N = 500;
  auto A = GenerateRandomMatrix(N);
  auto B = GenerateRandomMatrix(N);
  std::vector<double> C(N * N, 0.0);
  auto expected_output = MultiplyMatrices(A, B, N);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(expected_output[i], C[i], 1e-6);
  }
}

TEST(vavilov_v_cannon_seq, test_500) {
  constexpr size_t N = 500;
  std::vector<double> A(N * N, 1.0);
  std::vector<double> B(N * N, 1.0);
  std::vector<double> C(N * N, 0.0);
  std::vector<double> expected_output(N * N, N);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(expected_output[i], C[i], 1e-9);
  }
}

TEST(vavilov_v_cannon_seq, test_500_from_file) {
  std::string line;
  std::ifstream test_file("seq/vavilov_v_cannon/data/test.txt");
  size_t N = 0;
  if (test_file.is_open() && std::getline(test_file, line)) {
    count = std::stoi(line);
  }
  test_file.close();

  std::vector<double> A(N * N, 1.0);
  std::vector<double> B(N * N, 1.0);
  std::vector<double> C(N * N, 0.0);
  std::vector<double> expected_output(N * N, N);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(expected_output[i], C[i], 1e-9);
  }
}

TEST(vavilov_v_cannon_seq, test_invalid_size) {
  constexpr size_t N = 501;
  auto A = GenerateRandomMatrix(N);
  auto B = GenerateRandomMatrix(N);
  std::vector<double> C(N * N, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  EXPECT_FALSE(task_seq.Validation());
}

TEST(vavilov_v_cannon_seq, test_identity_matrix) {
  constexpr size_t N = 500;
  auto A = GenerateRandomMatrix(N);
  std::vector<double> I(N * N, 0.0);
  std::vector<double> C(N * N, 0.0);

  for (size_t i = 0; i < N; i++) {
    I[i * N + i] = 1.0;
  }

  auto expected_output = A;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(I.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(I.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(expected_output[i], C[i], 1e-6);
  }
}

TEST(vavilov_v_cannon_seq, test_zero_matrix) {
  constexpr size_t N = 500;
  auto A = GenerateRandomMatrix(N);
  std::vector<double> B(N * N, 0.0);
  std::vector<double> C(N * N, 0.0);
  std::vector<double> expected_output(N * N, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_EQ(expected_output[i], C[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_large_matrix) {
  constexpr size_t N = 1000;
  auto A = GenerateRandomMatrix(N, -1.0, 1.0);
  auto B = GenerateRandomMatrix(N, -1.0, 1.0);
  std::vector<double> C(N * N, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  ASSERT_NO_THROW(task_seq.Run());
  task_seq.PostProcessing();
}
