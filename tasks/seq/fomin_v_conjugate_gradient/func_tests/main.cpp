#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/fomin_v_conjugate_gradient/include/ops_seq.hpp"

TEST(conjugate_gradient_task_seq, test_small_system) {
  // Создаем данные для системы 3x3
  constexpr size_t kCount = 3;
  std::vector<double> A = {4, 1, 1, 1, 3, 0, 1, 0, 2};  // Матрица A
  std::vector<double> b = {6, 5, 3};                    // Вектор b
  std::vector<double> expected_x = {1, 1, 1};           // Ожидаемое решение
  std::vector<double> out(kCount, 0.0);                 // Выходной вектор

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(A.size() + b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], expected_x[i], 1e-6);
  }
}

TEST(conjugate_gradient_task_seq, test_large_system) {
  // Создаем данные для системы 5x5
  constexpr size_t kCount = 5;
  std::vector<double> A = {5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 5};  // Матрица A
  std::vector<double> b = {6, 7, 7, 7, 6};           // Вектор b
  std::vector<double> expected_x = {1, 1, 1, 1, 1};  // Ожидаемое решение
  std::vector<double> out(kCount, 0.0);              // Выходной вектор

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(A.size() + b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], expected_x[i], 1e-6);
  }
}

TEST(fomin_v_conjugate_gradient_seq, DotProduct) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  double expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;  // 32.0
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq task(task_data);

  EXPECT_DOUBLE_EQ(task.DotProduct(a, b), expected);
}

TEST(fomin_v_conjugate_gradient_seq, MatrixVectorMultiply) {
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> x = {5.0, 6.0};
  std::vector<double> expected = {1.0 * 5.0 + 2.0 * 6.0, 3.0 * 5.0 + 4.0 * 6.0};  // {17.0, 39.0}
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq task(task_data);

  auto result = task.MatrixVectorMultiply(A, x);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(fomin_v_conjugate_gradient_seq, VectorAdd) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {5.0, 7.0, 9.0};
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq task(task_data);

  auto result = task.VectorAdd(a, b);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(fomin_v_conjugate_gradient_seq, VectorSub) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  std::vector<double> expected = {-3.0, -3.0, -3.0};
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq task(task_data);

  auto result = task.VectorSub(a, b);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(fomin_v_conjugate_gradient_seq, VectorScalarMultiply) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  double scalar = 2.0;
  std::vector<double> expected = {2.0, 4.0, 6.0};
  auto task_data = std::make_shared<ppc::core::TaskData>();
  fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq task(task_data);

  auto result = task.VectorScalarMultiply(v, scalar);
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}
