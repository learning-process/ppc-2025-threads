#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

TEST(zinoviev_a_convex_hull_components_seq, test_square) {
  std::vector<int> input = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  std::vector<zinoviev_a_convex_hull_components_seq::Point> expected_output = {
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 4, .y = 4}, {.x = 0, .y = 4}};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(5);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_seq::Point[4]));
  task_data->outputs_count.emplace_back(4);

  zinoviev_a_convex_hull_components_seq::ConvexHullSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_seq::Point> actual_output(output, output + 4);

  ASSERT_EQ(actual_output.size(), expected_output.size());
  for (size_t i = 0; i < actual_output.size(); ++i) {
    ASSERT_EQ(actual_output[i].x, expected_output[i].x);
    ASSERT_EQ(actual_output[i].y, expected_output[i].y);
  }

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
}

namespace {
void VerifyPoints(const std::vector<zinoviev_a_convex_hull_components_seq::Point>& actual,
                  const std::vector<zinoviev_a_convex_hull_components_seq::Point>& expected) {
  for (const auto& expected_point : expected) {
    bool found = false;
    for (const auto& actual_point : actual) {
      if (actual_point.x == expected_point.x && actual_point.y == expected_point.y) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Point (" << expected_point.x << "," << expected_point.y << ") not found";
  }
}
}  // namespace

TEST(zinoviev_a_convex_hull_components_seq, test_triangle) {
  std::vector<int> input = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};

  std::vector<zinoviev_a_convex_hull_components_seq::Point> expected_output = {
      {.x = 0, .y = 0}, {.x = 0, .y = 4}, {.x = 2, .y = 2}};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(5);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_seq::Point[3]));
  task_data->outputs_count.emplace_back(3);

  zinoviev_a_convex_hull_components_seq::ConvexHullSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_seq::Point> actual_output(output, output + task_data->outputs_count[0]);

  ASSERT_EQ(actual_output.size(), expected_output.size());
  VerifyPoints(actual_output, expected_output);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
}