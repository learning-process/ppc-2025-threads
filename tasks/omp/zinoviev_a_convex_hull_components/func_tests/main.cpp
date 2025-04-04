#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/zinoviev_a_convex_hull_components/include/ops_omp.hpp"

namespace {
void SetupTest(std::shared_ptr<ppc::core::TaskData>& data, const std::vector<int>& input, int w, int h,
               size_t out_size) {
  data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  data->inputs_count.push_back(w);
  data->inputs_count.push_back(h);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[out_size]));
  data->outputs_count.push_back(static_cast<int>(out_size));
}

void CheckResult(const std::vector<zinoviev_a_convex_hull_components_omp::Point>& result,
                 const std::vector<zinoviev_a_convex_hull_components_omp::Point>& expect) {
  ASSERT_EQ(result.size(), expect.size());
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i].x, expect[i].x);
    ASSERT_EQ(result[i].y, expect[i].y);
  }
}

void RunTest(const std::vector<int>& input, const std::vector<zinoviev_a_convex_hull_components_omp::Point>& expect,
             int w, int h) {
  std::shared_ptr<ppc::core::TaskData> data;
  SetupTest(data, input, w, h, expect.size());

  zinoviev_a_convex_hull_components_omp::ConvexHullOMP task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* res = reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_omp::Point> actual(res, res + expect.size());
  CheckResult(actual, expect);
  delete[] res;
}
}  // namespace

TEST(zinoviev_a_convex_hull_omp, EmptyImage) {
  std::vector<int> input(25, 0);
  RunTest(input, {}, 5, 5);
}

TEST(zinoviev_a_convex_hull_omp, FullRectangle) {
  std::vector<int> input(25, 1);
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 3, .y = 4}, {.x = 0, .y = 4}};
  RunTest(input, expect, 5, 5);
}

TEST(zinoviev_a_convex_hull_omp, CrossShape) {
  std::vector<int> input = {0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_omp::Point> expect{
      {.x = 0, .y = 1}, {.x = 1, .y = 0}, {.x = 4, .y = 1}, {.x = 1, .y = 4}};
  RunTest(input, expect, 5, 5);
}