#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/shlyakov_m_shell_sort/include/ops_seq.hpp"

std::vector<int> static GenerateRandomArray(size_t size) {
  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = rand() % 100;
  }
  return arr;
}

bool is_sorted(const std::vector<int>& arr) {
  if (arr.empty()) {
    return true;
  }
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}

TEST(shlyakov_m_shell_sort_seq, Test_Empty_Array) {
  std::vector<int> in;
  std::vector<int> out;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_TRUE(is_sorted(out));
}

TEST(shlyakov_m_shell_sort_seq, Test_Already_Sorted_Array) {
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_TRUE(is_sorted(out));
  EXPECT_EQ(in, out);
}

TEST(shlyakov_m_shell_sort_seq, Test_Reverse_Sorted_Array) {
  std::vector<int> in = {5, 4, 3, 2, 1};
  std::vector<int> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_TRUE(is_sorted(out));
  std::vector<int> expected = {1, 2, 3, 4, 5};
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_seq, Test_Random_Array_Small) {
  std::vector<int> in = GenerateRandomArray(10);
  std::vector<int> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_TRUE(is_sorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected.begin(), expected.end());
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_seq, Test_Random_Array_Large) {
  size_t array_size = 200;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  ASSERT_TRUE(test_task_sequential.PreProcessing());
  ASSERT_TRUE(test_task_sequential.Run());
  ASSERT_TRUE(test_task_sequential.PostProcessing());

  EXPECT_TRUE(is_sorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected.begin(), expected.end());
  EXPECT_EQ(expected, out);
}