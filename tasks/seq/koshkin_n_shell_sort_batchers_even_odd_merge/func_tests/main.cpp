#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/koshkin_n_shell_sort_batchers_even_odd_merge/include/ops_seq.hpp"

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, positiveVectorAscending) {
  bool order = 1;

  std::vector<int> in = {34, 8, 64, 51, 32, 21, 99, 3, 45, 12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, negativeVectorAscending) {
  bool order = 1;

  std::vector<int> in = {-34, -8, -64, -51, -32, -21, -99, -3, -45, -12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, negativeVectorDescending) {
  bool order = 0;

  std::vector<int> in = {-34, -8, -64, -51, -32, -21, -99, -3, -45, -12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end(), std::greater<int>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, positiveVectorDescending) {
  bool order = 0;

  std::vector<int> in = {34, 8, 64, 51, 32, 21, 99, 3, 45, 12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end(), std::greater<int>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, smallVectorDescending) {
  bool order = 0;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::getRandomVector(15);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end(), std::greater<int>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, smallVectorAscending) {
  bool order = 1;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::getRandomVector(15);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, bigVectorDescending) {
  bool order = 0;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::getRandomVector(1500);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end(), std::greater<int>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, bigVectorAscending) {
  bool order = 1;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::getRandomVector(15);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res, out);
}