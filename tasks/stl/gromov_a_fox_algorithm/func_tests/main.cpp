#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/gromov_a_fox_algorithm/include/ops_stl.hpp"

namespace {
std::vector<double> NaiveMatrixMultiply(const std::vector<double>& a, const std::vector<double>& b, size_t n) {
  if (n == 0) {
    return {};
  }
  std::vector<double> result(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        result[(i * n) + j] += a[(i * n) + k] * b[(k * n) + j];
      }
    }
  }
  return result;
}

}  // namespace

TEST(gromov_a_fox_algorithm_stl, test_4x4) {
  constexpr size_t kN = 4;

  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> b = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = NaiveMatrixMultiply(a, b, kN);

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_stl, identity_3x3) {
  constexpr size_t kN = 3;

  std::vector<double> a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_stl, test_run_small_matrix_2x2_all_ones) {
  constexpr size_t kN = 2;

  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  std::vector<double> expected(kN * kN, static_cast<double>(kN));
  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_stl, test_zero_matrix_4x4) {
  constexpr size_t kN = 4;

  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected(kN * kN, 0.0);

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_stl, test_negative_values_3x3) {
  constexpr size_t kN = 3;

  std::vector<double> a = {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0};
  std::vector<double> b = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = NaiveMatrixMultiply(a, b, kN);

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_stl, identity_times_arbitrary_3x3) {
  constexpr size_t kN = 3;

  std::vector<double> a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = b;

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_stl, test_1x1_matrix) {
  constexpr size_t kN = 1;

  std::vector<double> a = {5.0};
  std::vector<double> b = {10.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = {50.0};

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  ASSERT_TRUE(test_task_stl.PreProcessing());
  ASSERT_TRUE(test_task_stl.Run());
  ASSERT_TRUE(test_task_stl.PostProcessing());

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}