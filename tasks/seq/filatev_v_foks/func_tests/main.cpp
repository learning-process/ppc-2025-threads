#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/filatev_v_foks/include/ops_seq.hpp"

namespace {

std::vector<double> GeneratMatrix(filatev_v_foks_seq::MatrixSize size);
std::vector<double> IdentityMatrix(size_t size);

std::vector<double> GeneratMatrix(filatev_v_foks_seq::MatrixSize size) {
  std::vector<double> matrix(size.n * size.m);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (auto &el : matrix) {
    el = dist(gen);
  }

  return matrix;
}

std::vector<double> IdentityMatrix(size_t size) {
  std::vector<double> matrix(size * size, 0);

  for (size_t i = 0; i < size; i++) {
    matrix[(i * size) + i] = 1;
  }

  return matrix;
}

}  // namespace

TEST(filatev_v_foks_seq, test_1) {
  filatev_v_foks_seq::MatrixSize size_a(4, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 4);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 2;

  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {8, 1, 1, 5, 4, 1, 0, 6, 5, 3, 1, 1, 1, 4, 2, 9};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.Run();
  focks.PostProcessing();

  std::vector<double> matrix_ans = {35, 28, 12, 56, 107, 64, 28, 140, 179, 100, 44, 224, 251, 136, 60, 308};

  EXPECT_EQ(matrix_ans, matrix_c);
}

TEST(filatev_v_foks_seq, test_2) {
  filatev_v_foks_seq::MatrixSize size_a(4, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 4);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 3;

  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {8, 1, 1, 5, 4, 1, 0, 6, 5, 3, 1, 1, 1, 4, 2, 9};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  std::vector<double> matrix_ans = {35, 28, 12, 56, 107, 64, 28, 140, 179, 100, 44, 224, 251, 136, 60, 308};

  EXPECT_EQ(matrix_ans, matrix_c);
}

TEST(filatev_v_foks_seq, test_3) {
  filatev_v_foks_seq::MatrixSize size_a(4, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 4);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 2;

  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  EXPECT_EQ(matrix_a, matrix_c);
}

TEST(filatev_v_foks_seq, test_10) {
  filatev_v_foks_seq::MatrixSize size_a(10, 10);
  filatev_v_foks_seq::MatrixSize size_b(10, 10);
  filatev_v_foks_seq::MatrixSize size_c(10, 10);

  size_t size_block = 2;

  std::vector<double> matrix_a = GeneratMatrix(size_a);
  std::vector<double> matrix_b = IdentityMatrix(size_b.n);
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  EXPECT_EQ(matrix_a, matrix_c);
}

TEST(filatev_v_foks_seq, test_10_5) {
  filatev_v_foks_seq::MatrixSize size_a(10, 10);
  filatev_v_foks_seq::MatrixSize size_b(10, 10);
  filatev_v_foks_seq::MatrixSize size_c(10, 10);

  size_t size_block = 5;

  std::vector<double> matrix_a = GeneratMatrix(size_a);
  std::vector<double> matrix_b = IdentityMatrix(size_b.n);
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  EXPECT_EQ(matrix_a, matrix_c);
}

TEST(filatev_v_foks_seq, test_4_3) {
  filatev_v_foks_seq::MatrixSize size_a(3, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 3);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 2;

  std::vector<double> matrix_a = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15};
  std::vector<double> matrix_b = {8, 1, 1, 5, 4, 1, 0, 6, 5, 3, 1, 1};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  std::vector<double> matrix_ans = {31, 12, 4, 20, 99, 32, 12, 68, 167, 52, 20, 116, 235, 72, 28, 164};

  EXPECT_EQ(matrix_ans, matrix_c);
}

TEST(filatev_v_foks_seq, test_4_1) {
  filatev_v_foks_seq::MatrixSize size_a(1, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 1);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 2;

  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), true);
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  std::vector<double> matrix_ans = {8, 1, 1, 5, 40, 5, 5, 25, 72, 9, 9, 45, 104, 13, 13, 65};

  EXPECT_EQ(matrix_ans, matrix_c);
}

TEST(filatev_v_foks_seq, test_error_1) {
  filatev_v_foks_seq::MatrixSize size_a(1, 4);
  filatev_v_foks_seq::MatrixSize size_b(1, 4);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 2;

  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), false);
}

TEST(filatev_v_foks_seq, test_error_2) {
  filatev_v_foks_seq::MatrixSize size_a(1, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 1);
  filatev_v_foks_seq::MatrixSize size_c(1, 1);

  size_t size_block = 2;

  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), false);
}

TEST(filatev_v_foks_seq, test_error_3) {
  filatev_v_foks_seq::MatrixSize size_a(1, 4);
  filatev_v_foks_seq::MatrixSize size_b(4, 1);
  filatev_v_foks_seq::MatrixSize size_c(4, 4);

  size_t size_block = 0;

  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(size_a.n);
  task_data->inputs_count.emplace_back(size_a.m);
  task_data->inputs_count.emplace_back(size_b.n);
  task_data->inputs_count.emplace_back(size_b.m);
  task_data->inputs_count.emplace_back(size_block);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

  task_data->outputs_count.emplace_back(size_c.n);
  task_data->outputs_count.emplace_back(size_c.m);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));

  filatev_v_foks_seq::Focks focks(task_data);
  ASSERT_EQ(focks.Validation(), false);
}