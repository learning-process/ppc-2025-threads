#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/zinoviev_a_convex_hull_components/include/ops_tbb.hpp"

namespace {
void SetupTaskData(const std::vector<int>& input, int width, int height, size_t output_size,
                   std::shared_ptr<ppc::core::TaskData>& task_data) {
  task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(
      reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_tbb::Point[output_size]));
  task_data->outputs_count.emplace_back(output_size);
}

void VerifyResult(const std::vector<zinoviev_a_convex_hull_components_tbb::Point>& actual,
                  const std::vector<zinoviev_a_convex_hull_components_tbb::Point>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].x, expected[i].x);
    ASSERT_EQ(actual[i].y, expected[i].y);
  }
}

void RunAndValidate(const std::vector<int>& input,
                    const std::vector<zinoviev_a_convex_hull_components_tbb::Point>& expected, int width, int height) {
  std::shared_ptr<ppc::core::TaskData> task_data;
  SetupTaskData(input, width, height, expected.size(), task_data);

  zinoviev_a_convex_hull_components_tbb::ConvexHullTBB task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_tbb::Point*>(task_data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_tbb::Point> actual(output, output + expected.size());

  VerifyResult(actual, expected);
  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_tbb::Point*>(task_data->outputs[0]);
}
}  // namespace

TEST(zinoviev_a_convex_hull_components_tbb, SquareShape) {
  const std::vector<int> input = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_tbb::Point> expected = {
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 3, .y = 4}, {.x = 0, .y = 4}};
  RunAndValidate(input, expected, 5, 5);
}

TEST(zinoviev_a_convex_hull_components_tbb, TriangleShape) {
  const std::vector<int> input = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_tbb::Point> expected = {
      {.x = 0, .y = 0}, {.x = 2, .y = 2}, {.x = 0, .y = 4}};
  RunAndValidate(input, expected, 5, 5);
}