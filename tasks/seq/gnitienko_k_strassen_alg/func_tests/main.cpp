#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/gnitienko_k_strassen_alg/include/ops_seq.hpp"

namespace gnitienko_k_matrix_func {
static double min_val = -50.0;
static double max_val = 50.0;
static std::vector<double> GenMatrix(size_t size);
static void TrivialMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                            size_t size);

std::vector<double> GenMatrix(size_t size) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}

void TrivialMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      c[(i * size) + j] = 0;
      for (size_t k = 0; k < size; ++k) {
        c[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
        c[(i * size) + j] = round(c[(i * size) + j] * 10000) / 10000;
      }
    }
  }
}
}  // namespace gnitienko_k_matrix_func

TEST(gnitienko_k_strassen_alg_seq, test_2x2_matrix) {
  // Create data
  std::vector<double> a = {2.4, 3.5, -4.1, 13.3};
  std::vector<double> b = {1.4, -0.5, 1.1, 2.3};
  std::vector<double> expected = {7.21, 6.85, 8.89, 32.64};
  std::vector<double> out(4);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  gnitienko_k_strassen_algorithm::StrassenAlgSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected, out);
}

TEST(gnitienko_k_strassen_alg_seq, test_4x4_matrix) {
  // Create data
  std::vector<double> a = {2.4, 3.5, -4.1, 13.3, 1.4, -0.5, 1.1, 2.3, 3.2, 2.1, -1.3, 4.5, 0.9, -2.7, 3.8, -1.2};
  std::vector<double> b = {1.1, -0.8, 2.3, 0.5, -1.5, 3.2, 0.7, 1.9, 0.9, -1.1, 1.5, -0.4, 2.2, 0.6, -3.1, 1.3};
  std::vector<double> expected = {22.96, 21.77, -39.41, 26.78, 8.34, -2.55,  -2.61, 2.3,
                                  9.1,   8.29,  -7.07,  11.96, 5.82, -14.26, 9.6,   -7.76};
  std::vector<double> out(16);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  gnitienko_k_strassen_algorithm::StrassenAlgSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected, out);
}

TEST(gnitienko_k_strassen_alg_seq, test_random_16x16) {
  // Create data
  size_t size = 16;
  std::vector<double> a = gnitienko_k_matrix_func::GenMatrix(size);
  std::vector<double> b = gnitienko_k_matrix_func::GenMatrix(size);
  std::vector<double> expected(size * size);
  gnitienko_k_matrix_func::TrivialMultiply(a, b, expected, size);
  std::vector<double> out(size * size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  gnitienko_k_strassen_algorithm::StrassenAlgSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < size * size; i++) {
    EXPECT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(gnitienko_k_strassen_alg_seq, test_non_squad_7x7) {
  // Create data
  size_t size = 7;
  std::vector<double> a = gnitienko_k_matrix_func::GenMatrix(size);
  std::vector<double> b = gnitienko_k_matrix_func::GenMatrix(size);
  std::vector<double> expected(size * size);
  gnitienko_k_matrix_func::TrivialMultiply(a, b, expected, size);
  std::vector<double> out(size * size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  gnitienko_k_strassen_algorithm::StrassenAlgSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < size * size; i++) {
    EXPECT_NEAR(expected[i], out[i], 1e-3);
  }
}

TEST(gnitienko_k_strassen_alg_seq, test_empty) {
  // Create data
  std::vector<double> a = {};
  std::vector<double> b = {};
  std::vector<double> expected = {};
  std::vector<double> out = {};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  gnitienko_k_strassen_algorithm::StrassenAlgSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected, out);
}