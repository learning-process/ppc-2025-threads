#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/vershinina_a_hoare_sort/include/ops_seq.hpp"

std::vector<int> getRandomVector(int len) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  std::vector<int> vec(len);
  int vec_size = vec.size();
  for (int i = 0; i < vec_size; i++) {
    vec[i] = distr(gen);
  }
  return vec;
}

TEST(vershinina_a_hoare_sort, Test_empty_seq) {
  std::vector<int> in;
  std::vector<int> out;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  vershinina_a_hoare_sort::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  ASSERT_EQ(in, out);
}

TEST(vershinina_a_hoare_sort, Test_len_20_seq) {
  std::vector<int> in;
  std::vector<int> out(20);
  in = getRandomVector(20);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  ASSERT_EQ(in, out);
}

TEST(vershinina_a_hoare_sort, Test_len_50_seq) {
  std::vector<int> in;
  std::vector<int> out(50);
  in = getRandomVector(50);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  ASSERT_EQ(in, out);
}

TEST(vershinina_a_hoare_sort, Test_len_100_seq) {
  std::vector<int> in;
  std::vector<int> out(100);
  in = getRandomVector(100);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  ASSERT_EQ(in, out);
}

TEST(vershinina_a_hoare_sort, Test_len_200_seq) {
  std::vector<int> in;
  std::vector<int> out(200);
  in = getRandomVector(200);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  vershinina_a_hoare_sort::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  ASSERT_EQ(in, out);
}