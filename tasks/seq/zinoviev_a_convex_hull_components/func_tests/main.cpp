#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

namespace {
void setup_task_data(const std::vector<int>& input, int width, int height, size_t output_size,
                     std::shared_ptr<ppc::core::TaskData>& task_data) {
  task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(
      reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_seq::Point[output_size]));
  task_data->outputs_count.emplace_back(output_size);
}

void verify_result(const std::vector<zinoviev_a_convex_hull_components_seq::Point>& actual,
                   const std::vector<zinoviev_a_convex_hull_components_seq::Point>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].x, expected[i].x);
    ASSERT_EQ(actual[i].y, expected[i].y);
  }
}

void run_and_validate(const std::vector<int>& input,
                      const std::vector<zinoviev_a_convex_hull_components_seq::Point>& expected, int width,
                      int height) {
  std::shared_ptr<ppc::core::TaskData> task_data;
  setup_task_data(input, width, height, expected.size(), task_data);

  zinoviev_a_convex_hull_components_seq::ConvexHullSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_seq::Point> actual(output, output + expected.size());

  verify_result(actual, expected);
  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_seq::Point*>(task_data->outputs[0]);
}
}  // namespace

TEST(zinoviev_a_convex_hull_components_seq, SquareShape) {
  constexpr int k_width = 5;
  constexpr int k_height = 5;
  const std::vector<int> input = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 3, .y = 4}, {.x = 0, .y = 4}};
  run_and_validate(input, expected, k_width, k_height);
}

TEST(zinoviev_a_convex_hull_components_seq, TriangleShape) {
  constexpr int k_width = 5;
  constexpr int k_height = 5;
  const std::vector<int> input = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {
      {.x = 0, .y = 0}, {.x = 2, .y = 2}, {.x = 0, .y = 4}};
  run_and_validate(input, expected, k_width, k_height);
}

TEST(zinoviev_a_convex_hull_components_seq, LargeRectangle) {
  constexpr int k_width = 20;
  constexpr int k_height = 10;
  std::vector<int> input(k_width * k_height, 0);

  for (int x = 0; x < k_width; ++x) {
    input[x] = 1;
    input[((k_height - 1) * k_width) + x] = 1;
  }

  for (int y = 0; y < k_height; ++y) {
    input[y * k_width] = 1;
    input[(y * k_width) + (k_width - 1)] = 1;
  }

  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {
      {.x = 0, .y = 0}, {.x = 19, .y = 0}, {.x = 18, .y = 9}, {.x = 0, .y = 9}};
  run_and_validate(input, expected, k_width, k_height);
}

TEST(zinoviev_a_convex_hull_components_seq, SinglePoint) {
  constexpr int k_width = 1;
  constexpr int k_height = 1;
  const std::vector<int> input = {1};
  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {{.x = 0, .y = 0}};
  run_and_validate(input, expected, k_width, k_height);
}

TEST(zinoviev_a_convex_hull_components_seq, SmallGrid) {
  constexpr int k_width = 4;
  constexpr int k_height = 4;
  const std::vector<int> input = {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_seq::Point> expected = {
      {.x = 0, .y = 0}, {.x = 3, .y = 0}, {.x = 2, .y = 3}, {.x = 0, .y = 3}};
  run_and_validate(input, expected, k_width, k_height);
}