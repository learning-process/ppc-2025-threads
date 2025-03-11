#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/ermolaev_v_graham_scan/include/ops_seq.hpp"

namespace {
ppc::core::TaskDataPtr CreateTaskData(std::vector<ermolaev_v_graham_scan_seq::Point>& input,
                                      std::vector<ermolaev_v_graham_scan_seq::Point>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  return task_data;
}

void RunTest(ppc::core::TaskDataPtr& task_data) {
  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
}  // namespace

TEST(ermolaev_v_graham_scan_seq, triangle) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {2, 2}, {2, 0}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  RunTest(task_data);

  EXPECT_EQ(task_data->outputs_count[0], 3U);
  auto* result = reinterpret_cast<ermolaev_v_graham_scan_seq::Point*>(task_data->outputs[0]);
  std::vector<ermolaev_v_graham_scan_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}};
  EXPECT_EQ(std::vector<ermolaev_v_graham_scan_seq::Point>(result, result + 3), expected);
}

TEST(ermolaev_v_graham_scan_seq, square_with_inner_point) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {0, 2}, {2, 0}, {2, 2}, {1, 1}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  RunTest(task_data);

  EXPECT_EQ(task_data->outputs_count[0], 4U);
}

TEST(ermolaev_v_graham_scan_seq, all_same_points) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
}

TEST(ermolaev_v_graham_scan_seq, collinear_points) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
}

TEST(ermolaev_v_graham_scan_seq, collinear_points_x) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
}

TEST(ermolaev_v_graham_scan_seq, collinear_points_y) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
}

TEST(ermolaev_v_graham_scan_seq, invalid_input) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input = {{0, 0}, {0, 0}};
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(ermolaev_v_graham_scan_seq, empty_input) {
  std::vector<ermolaev_v_graham_scan_seq::Point> input;
  std::vector<ermolaev_v_graham_scan_seq::Point> output(input.size());
  auto task_data = CreateTaskData(input, output);

  ermolaev_v_graham_scan_seq::TestTaskSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(ermolaev_v_graham_scan_seq, random_points) {
  constexpr int kCount = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-100, 100);

  std::vector<ermolaev_v_graham_scan_seq::Point> input;
  std::vector<ermolaev_v_graham_scan_seq::Point> output(kCount);
  input.reserve(kCount);
  for (int i = 0; i < kCount; ++i) {
    input.emplace_back(dis(gen), dis(gen));
  }

  auto task_data = CreateTaskData(input, output);
  RunTest(task_data);

  auto* result = reinterpret_cast<ermolaev_v_graham_scan_seq::Point*>(task_data->outputs[0]);
  size_t hull_size = task_data->outputs_count[0];

  ermolaev_v_graham_scan_seq::Point p1;
  ermolaev_v_graham_scan_seq::Point p2;
  for (const auto& point : input) {
    for (size_t i = 0; i < hull_size; ++i) {
      p1 = result[i];
      p2 = result[(i + 1) % hull_size];

      int cross = ((p2.x - p1.x) * (point.y - p1.y)) - ((point.x - p1.x) * (p2.y - p1.y));
      EXPECT_GE(cross, 0);
    }
  }
}