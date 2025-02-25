#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

namespace vavilov_v_cannon_seq {

std::vector<double> GenerateRandomMatrix(unsigned int n, double min_val, double max_val) {
  std::vector<double> matrix(n * n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (unsigned int i = 0; i < n * n; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

std::vector<double> MultMat(const std::vector<double>& a, const std::vector<double>& b, unsigned int n) {
  std::vector<double> c(n * n, 0.0);
  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      for (unsigned int k = 0; k < n; k++) {
        c[(i * n) + j] += a[(i * n) + k] * b[(k * n) + j];
      }
    }
  }
  return c;
}

}  // namespace vavilov_v_cannon_seq

TEST(vavilov_v_cannon_seq, test_random) {
  constexpr unsigned int kN = 16;
  auto a = vavilov_v_cannon_seq::GenerateRandomMatrix(kN);
  auto b = vavilov_v_cannon_seq::GenerateRandomMatrix(kN);
  std::vector<double> expected_output = vavilov_v_cannon_seq::MultMat(a, b, kN);
  std::vector<double> c(kN * kN, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < kN * kN; i++) {
    EXPECT_NEAR(expected_output[i], c[i], 1e-6);
  }
}

TEST(vavilov_v_cannon_seq, test_fixed_4x4) {
  constexpr unsigned int kN = 4;
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> b = {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
  std::vector<double> expected_output = {4, 6, 6, 4, 12, 14, 14, 12, 20, 22, 22, 20, 28, 30, 30, 28};
  std::vector<double> c(kN * kN, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_invalid_size_1) {
  constexpr unsigned int kN = 51;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> c(kN * kN, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_FALSE(task_seq.Validation());
}

TEST(vavilov_v_cannon_seq, test_invalid_size_2) {
  std::vector<double> a(2 * 2, 1.0);
  std::vector<double> b(3 * 2, 1.0);
  std::vector<double> c(2 * 2, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_FALSE(task_seq.Validation());
}

TEST(vavilov_v_cannon_seq, test_225) {
  constexpr unsigned int kN = 225;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> c(kN * kN, 0.0);
  std::vector<double> expected_output(kN * kN, kN);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_225_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/vavilov_v_cannon/data/test.TXT"));
  unsigned int k_n = 0;
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();

  k_n = std::stoi(line);

  std::vector<double> a(k_n * k_n, 1.0);
  std::vector<double> b(k_n * k_n, 1.0);
  std::vector<double> c(k_n * k_n, 0.0);
  std::vector<double> expected_output(k_n * k_n, k_n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < k_n * k_n; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_identity_matrix) {
  constexpr unsigned int kN = 225;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> c(kN * kN, 0.0);

  for (unsigned int i = 0; i < kN; i++) {
    b[(i * kN) + i] = 1.0;
  }

  auto expected_output = a;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_zero_matrix) {
  constexpr unsigned int kN = 225;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> c(kN * kN, 0.0);
  std::vector<double> expected_output(kN * kN, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_seq->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}
