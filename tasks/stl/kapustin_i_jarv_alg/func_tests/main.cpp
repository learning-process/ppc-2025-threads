#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/kapustin_i_jarv_alg/include/ops_stl.hpp"

namespace {
std::vector<std::pair<int, int>> GenerateRandomPoints(size_t count, int min_x, int max_x, int min_y, int max_y) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist_x(min_x, max_x);
  std::uniform_int_distribution<int> dist_y(min_y, max_y);

  std::vector<std::pair<int, int>> random_points;
  random_points.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    random_points.emplace_back(dist_x(rng), dist_y(rng));
  }

  return random_points;
}
}  // namespace

TEST(KapustinJarvAlgSTLTest, SimpleTriangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 5}, {10, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 5}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, FixedPointsWithRandomNoise) {
  std::vector<std::pair<int, int>> fixed_points = {{-1000, -1000}, {1000, -1000}, {1000, 1000}, {-1000, 1000}};

  auto random_points = GenerateRandomPoints(100, -100, 100, -100, 100);

  std::vector<std::pair<int, int>> input_points = fixed_points;
  input_points.insert(input_points.end(), random_points.begin(), random_points.end());

  std::vector<std::pair<int, int>> output_result(fixed_points.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  EXPECT_EQ(output_result.size(), fixed_points.size());
  for (size_t i = 0; i < fixed_points.size(); ++i) {
    EXPECT_EQ(output_result[i], fixed_points[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, TriangleWithInnerPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}, {5, 4}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, LineOfPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {1, 0}, {2, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {2, 0}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, Rectangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 2}, {2, 0}, {2, 2}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, DuplicatePoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 0}, {10, 0}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, Star4Points) {
  std::vector<std::pair<int, int>> input_points = {{5, 10}, {5, 0}, {0, 5}, {10, 5}, {5, 5}};
  std::vector<std::pair<int, int>> expected_result = {{0, 5}, {5, 0}, {10, 5}, {5, 10}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSTLTest, Circle) {
  std::vector<std::pair<int, int>> input_points = {{1, 2}, {2, 1}, {2, 3}, {3, 2}};
  std::vector<std::pair<int, int>> expected_result = {{1, 2}, {2, 1}, {3, 2}, {2, 3}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_stl->inputs_count.emplace_back(input_points.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_stl->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}
