#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

namespace {
void VerifyConvexHull(const std::vector<int>& input,
                      const std::vector<zinoviev_a_convex_hull_components_seq::Point>& expected, int width,
                      int height) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(
      reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_seq::Point[expected.size()]));
  task_data->outputs_count.emplace_back(expected.size());

  zinoviev_a_convex_hull_components_seq::ConvexHullSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_seq::Point> actual(output, output + expected.size());

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].x, expected[i].x);
    ASSERT_EQ(actual[i].y, expected[i].y);
  }

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
}
}

TEST(ConvexHullTest, Square) {
  const std::vector<int> input = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 4, .y = 4}, {.x = 0, .y = 4}};
  VerifyConvexHull(input, expected, 5, 5);
}

TEST(ConvexHullTest, Triangle) {
  const std::vector<int> input = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {
      {.x = 0, .y = 0}, {.x = 0, .y = 4}, {.x = 2, .y = 2}};
  VerifyConvexHull(input, expected, 5, 5);
}