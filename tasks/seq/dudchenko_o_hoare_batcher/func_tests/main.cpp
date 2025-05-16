#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/dudchenko_o_hoare_batcher/include/ops_seq.hpp"

TEST(dudchenko_o_hoare_batcher_seq, test_sort_50) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount);
  std::vector<int> out(kCount);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (size_t i = 0; i < kCount; ++i) {
    in[i] = std::rand() % 1000;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  dudchenko_o_hoare_batcher_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> sorted_in = in;
  std::ranges::sort(sorted_in);
  EXPECT_EQ(sorted_in, out);
}

TEST(dudchenko_o_hoare_batcher_seq, test_sort_100_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/dudchenko_o_hoare_batcher/data/test.txt"));
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();

  const size_t count = std::stoi(line);

  std::vector<int> in(count);
  std::vector<int> out(count);

  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (size_t i = 0; i < count; ++i) {
    in[i] = std::rand() % 1000;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  dudchenko_o_hoare_batcher_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> sorted_in = in;
  std::ranges::sort(sorted_in);
  EXPECT_EQ(sorted_in, out);
}