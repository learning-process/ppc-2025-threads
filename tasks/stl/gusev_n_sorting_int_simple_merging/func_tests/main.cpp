#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/gusev_n_sorting_int_simple_merging/include/ops_stl.hpp"

namespace {
ppc::core::TaskDataPtr CreateTaskData(std::vector<int>& input, std::vector<int>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  return task_data;
}

void RunTest(std::vector<int>& input, std::vector<int>& output) {
  auto task_data = CreateTaskData(input, output);
  gusev_n_sorting_int_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}
}  // namespace

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_basic) {
  std::vector<int> in = {170, 45, 75, 90, 802, 24, 2, 66};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_empty) {
  std::vector<int> in;
  std::vector<int> out;
  RunTest(in, out);
  EXPECT_TRUE(out.empty());
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_single_element) {
  std::vector<int> in = {42};
  std::vector<int> out(1);
  RunTest(in, out);
  EXPECT_EQ(in, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_negative_numbers) {
  std::vector<int> in = {3, -1, 0, -5, 2, -3};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_random) {
  std::vector<int> in(100);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10000, 10000);
  for (int& num : in) {
    num = dist(gen);
  }
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_duplicates) {
  std::vector<int> in = {5, 3, 5, -2, 3, -2, -2, 5, 0, 0, 7, 7, -7};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_stl, test_radix_sort_reverse_order) {
  std::vector<int> in = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3};
  std::vector<int> out(in.size());
  RunTest(in, out);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}
