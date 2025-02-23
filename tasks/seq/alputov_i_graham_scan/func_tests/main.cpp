#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/alputov_i_graham_scan/include/ops_seq.hpp"

namespace {
void GenerateRandomData(std::vector<alputov_i_graham_scan_seq::Point>& data, size_t count) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  data.clear();
  data.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    data.emplace_back(dist(gen), dist(gen));
  }
}
}  // namespace

TEST(alputov_i_graham_scan_seq, minimal_triangle_case) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {2, 0}, {1, 2}};
  std::vector<alputov_i_graham_scan_seq::Point> output(3);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetConvexHull().size(), 3U);
}

TEST(alputov_i_graham_scan_seq, collinear_points) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 2U);
  bool order1 = (convex_hull[0].x == 0 && convex_hull[1].x == 3);
  bool order2 = (convex_hull[0].x == 3 && convex_hull[1].x == 0);
  EXPECT_TRUE(order1 || order2);
}

TEST(alputov_i_graham_scan_seq, perfect_square_case) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 4U);
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  for (const auto& p : input) {
    EXPECT_TRUE(hull_set.count(p));
  }
}

TEST(alputov_i_graham_scan_seq, random_1000_points) {
  std::vector<alputov_i_graham_scan_seq::Point> input;
  GenerateRandomData(input, 1000);
  input.insert(input.end(), {{-1500, -1500}, {1500, -1500}, {1500, 1500}, {-1500, 1500}});

  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto& convex_hull = task.GetConvexHull();
  auto contains = [&](double x, double y) {
    return std::any_of(convex_hull.begin(), convex_hull.end(), [x, y](const auto& p) { return p.x == x && p.y == y; });
  };

  EXPECT_TRUE(contains(-1500, -1500));
  EXPECT_TRUE(contains(1500, 1500));
  EXPECT_LE(convex_hull.size(), input.size());
}

TEST(alputov_i_graham_scan_seq, duplicate_points) {
  std::vector<alputov_i_graham_scan_seq::Point> input(10, {2.5, 3.5});
  input.insert(input.end(), {{0, 0}, {5, 0}, {5, 5}});

  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto& convex_hull = task.GetConvexHull();
  std::set<alputov_i_graham_scan_seq::Point> unique_hull(convex_hull.begin(), convex_hull.end());
  EXPECT_EQ(unique_hull.size(), 4U);
}
