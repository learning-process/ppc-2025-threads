#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/shuravina_o_hoare_simple_merger_all1/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

bool IsSorted(const std::vector<int>& arr) {
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> GenerateRandomVector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-1000, 1000);

  std::vector<int> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = distrib(gen);
  }
  return vec;
}

}  // namespace

TEST(shuravina_o_hoare_simple_merger_all, test_random_array) {
  const size_t size = 1000;
  std::vector<int> input = GenerateRandomVector(size);
  std::vector<int> output(size, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  shuravina_o_hoare_simple_merger::TestTaskALL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_TRUE(IsSorted(output));
}

TEST(shuravina_o_hoare_simple_merger_all, test_empty_array) {
  std::vector<int> input;
  std::vector<int> output;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(0);
  task_data->outputs.emplace_back(nullptr);
  task_data->outputs_count.emplace_back(0);

  shuravina_o_hoare_simple_merger::TestTaskALL task(task_data);
  EXPECT_FALSE(task.Validation());
}