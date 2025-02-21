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
#include "seq/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_seq.hpp"

namespace volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq {
void GetRandomVector(std::vector<int> &v, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dis(gen);
  }
}
}  // namespace volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_error_in_val) {
  constexpr size_t size_of_vector = 0;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  std::vector<int> out(size_of_vector, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector) {
  constexpr size_t size_of_vector = 100;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector2) {
  constexpr size_t size_of_vector = 200;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector3) {
  constexpr size_t size_of_vector = 300;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector4) {
  constexpr size_t size_of_vector = 400;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector) {
  constexpr size_t size_of_vector = 500;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector2) {
  constexpr size_t size_of_vector = 600;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector3) {
  constexpr size_t size_of_vector = 700;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector4) {
  constexpr size_t size_of_vector = 800;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector5) {
  constexpr size_t size_of_vector = 900;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector) {
  constexpr size_t size_of_vector = 1000;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector2) {
  constexpr size_t size_of_vector = 2000;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector3) {
  constexpr size_t size_of_vector = 3000;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector4) {
  constexpr size_t size_of_vector = 4000;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_extra_big_vector) {
  constexpr size_t size_of_vector = 10000;

  // Create data
  std::vector<int> in(size_of_vector, 0);
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::GetRandomVector(in, -100, 100);
  std::vector<int> out(size_of_vector, 0);
  std::vector<int> answer(in);
  std::sort(answer.begin(), answer.end());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}
