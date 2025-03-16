#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq/include/ops_seq.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq, test_small_array) {
  std::vector<int> input = {35, 33, 42, 10, 14, 19, 27, 44};
  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq, test_small_array_with_negative_values) {
  std::vector<int> input = {35, 33, 42, -10, -14, 19, 27, 44};
  std::vector<int> expected_output = {-14, -10, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq, test_empty_array) {
  std::vector<int> input = {};
  std::vector<int> expected_output = {};

  std::vector<int> output(input.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output, expected_output);
}