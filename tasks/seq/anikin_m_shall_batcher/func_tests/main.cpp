// Copyright Anikin Maksim 2025W
#include <gtest/gtest.h>

#include <random>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/anikin_m_shall_batcher/include/ops_seq.hpp"

namespace {
void FillVectorWithRandomValues(std::vector<int>& vec, int size, int min_val = 0, int max_val = 100) {
  vec.resize(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(min_val, max_val);

  for (int i = 0; i < size; ++i) {
    vec[i] = dist(gen);
  }
}
}  // namespace

TEST(anikin_m_shall_batcher_seq, test_sort_5) {
  // Create data
  int vec_size = 5;
  std::vector<int> in(vec_size, 0);
  std::vector<int> out(vec_size, 0);

  in[0] = 5;
  in[1] = -2;
  in[2] = 8;
  in[3] = 6;
  in[4] = 5;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_shall_batcher_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in.size(), out.size());
  for (int i = 1; i < vec_size; i++) {
    EXPECT_GE(out[i], out[i - 1]);
  }
}

TEST(anikin_m_shall_batcher_seq, test_sort_rand_100) {
  // Create data
  int vec_size = 100;
  std::vector<int> in(vec_size, 0);
  std::vector<int> out(vec_size, 0);

  FillVectorWithRandomValues(in, vec_size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_shall_batcher_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in.size(), out.size());
  for (int i = 1; i < vec_size; i++) {
    EXPECT_GE(out[i], out[i - 1]);
  }
}

TEST(anikin_m_shall_batcher_seq, test_sort_rand_1000000) {
  // Create data
  int vec_size = 1000000;
  std::vector<int> in(vec_size, 0);
  std::vector<int> out(vec_size, 0);

  FillVectorWithRandomValues(in, vec_size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_shall_batcher_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in.size(), out.size());
  for (int i = 1; i < vec_size; i++) {
    EXPECT_GE(out[i], out[i - 1]);
  }
}

TEST(anikin_m_shall_batcher_seq, size_0) {
  // Create data
  int vec_size = 0;
  std::vector<int> in(vec_size, 0);
  std::vector<int> out(vec_size, 0);

  FillVectorWithRandomValues(in, vec_size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_shall_batcher_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in.size(), out.size());
  for (int i = 1; i < vec_size; i++) {
    EXPECT_GE(out[i], out[i - 1]);
  }
}