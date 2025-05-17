#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/shuravina_o_hoare_simple_merger_all/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

bool IsReverseSorted(const std::vector<int>& arr) {
  if (arr.empty()) {
    return true;
  }
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] < arr[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> GenerateRandomArray(size_t size, int min_val, int max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(min_val, max_val);

  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = distrib(gen);
  }
  return arr;
}

}  // namespace

TEST(shuravina_o_hoare_simple_merger_all, test_random_array) {
  const size_t array_size = 1000;
  const int min_val = -1000;
  const int max_val = 1000;

  std::vector<int> in = GenerateRandomArray(array_size, min_val, max_val);
  std::vector<int> out(in.size(), 0);

  ASSERT_FALSE(IsReverseSorted(in));

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(shuravina_o_hoare_simple_merger_all, test_already_sorted_array) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  std::vector<int> out(in.size(), 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  std::vector<int> expected = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger_all, test_single_element_array) {
  std::vector<int> in = {42};
  std::vector<int> out(in.size(), 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  std::vector<int> expected = {42};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger_all, test_array_with_negative_numbers) {
  std::vector<int> in = {-5, 2, -9, 1, 0, -3};
  std::vector<int> out(in.size(), 0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  std::vector<int> expected = {-9, -5, -3, 0, 1, 2};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger_all, validation_empty_input_output) {
  auto empty_task_data = std::make_shared<ppc::core::TaskData>();
  empty_task_data->inputs.emplace_back(nullptr);
  empty_task_data->inputs_count.emplace_back(0);
  empty_task_data->outputs.emplace_back(nullptr);
  empty_task_data->outputs_count.emplace_back(0);

  shuravina_o_hoare_simple_merger::TestTaskALL empty_test_task(empty_task_data);
  EXPECT_FALSE(empty_test_task.Validation());
}

TEST(shuravina_o_hoare_simple_merger_all, validation_different_sizes) {
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out(2, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskALL test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(shuravina_o_hoare_simple_merger_all, validation_null_input_pointer) {
  std::vector<int> out(5, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(5);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskALL test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}

TEST(shuravina_o_hoare_simple_merger_all, validation_null_output_pointer) {
  std::vector<int> in = {1, 2, 3, 4, 5};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(nullptr);
  task_data->outputs_count.emplace_back(5);

  shuravina_o_hoare_simple_merger::TestTaskALL test_task(task_data);
  EXPECT_FALSE(test_task.Validation());
}