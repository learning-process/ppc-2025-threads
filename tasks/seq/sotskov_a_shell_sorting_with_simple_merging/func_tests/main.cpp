#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sotskov_a_shell_sorting_with_simple_merging/include/ops_seq.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_seq {

std::vector<int> GenerateRandomVector(const RandomVectorParams &params) {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<int> distribution(params.min_value, params.max_value);

  std::vector<int> random_vector(params.size);
  for (int &element : random_vector) {
    element = distribution(generator);
  }

  return random_vector;
}

void RunSortingTest(SortingTestParams &params, void (*sort_func)(std::vector<int> &)) {
  std::vector<int> out(params.input.size(), 0);

  std::vector<int> sorted_expected = params.expected;
  std::ranges::sort(sorted_expected.begin(), sorted_expected.end());

  sort_func(params.input);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(params.input.data()));
  task_data_seq->inputs_count.emplace_back(params.input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(out, sorted_expected);
}
}  // namespace sotskov_a_shell_sorting_with_simple_merging_seq

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_positive_numbers) {
  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams params = {.expected = {1, 2, 3, 4, 5, 6, 7, 8},
                                                                               .input = {5, 3, 8, 6, 2, 7, 1, 4}};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_negative_numbers) {
  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams params = {.expected = {-12, -10, -8, -7, -4, -3},
                                                                               .input = {-8, -3, -12, -7, -4, -10}};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_ordered_array) {
  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams params = {.expected = {1, 2, 3, 4, 5, 6, 7, 8},
                                                                               .input = {1, 2, 3, 4, 5, 6, 7, 8}};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_with_duplicates) {
  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams params = {.expected = {2, 2, 2, 4, 4, 6, 6, 8},
                                                                               .input = {4, 2, 2, 8, 4, 6, 6, 2}};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_single_element) {
  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams params = {.expected = {77}, .input = {77}};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_empty_array) {
  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams params = {.expected = {}, .input = {}};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_seq, test_sort_random_vector) {
  sotskov_a_shell_sorting_with_simple_merging_seq::RandomVectorParams params = {
      .size = 20, .min_value = -100, .max_value = 100};
  std::vector<int> in = sotskov_a_shell_sorting_with_simple_merging_seq::GenerateRandomVector(params);
  std::vector<int> expected = in;

  std::ranges::sort(expected.begin(), expected.end());

  sotskov_a_shell_sorting_with_simple_merging_seq::SortingTestParams sorting_params = {.expected = expected,
                                                                                       .input = in};

  sotskov_a_shell_sorting_with_simple_merging_seq::RunSortingTest(
      sorting_params, sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging);
}
