#include <gtest/gtest.h>
#include <limits.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/solovyev_d_shell_sort_simple/include/ops_seq.hpp"

namespace solovyev_d_shell_sort_simple_seq {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
bool isSorted(std::vector<int> data) {
  int last = INT_MIN;
  for (int i = 0; i < data.size(); i++) {
    if (data[i] < last) {
      return false;
    }
    last = data[i];
  }
  return true;
}
}  // namespace solovyev_d_shell_sort_simple_seq

TEST(solovyev_d_shell_sort_simple_seq, sort_10) {
  // Create data
  std::vector<int> in = {1, 5, 7, 3, 7, 3, 8, 4, 1, 6};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();
  ASSERT_TRUE(solovyev_d_shell_sort_simple_seq::isSorted(out));
}

TEST(solovyev_d_shell_sort_simple_seq, sort_20) {
  // Create data
  std::vector<int> in = {1, 5, 7, 3, 7, 3, 8, 4, 1, 6, 4, 6, 7, 3, 12, 21, 65, 43, 1, 54, 34, 76};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();
  ASSERT_TRUE(solovyev_d_shell_sort_simple_seq::isSorted(out));
}

TEST(solovyev_d_shell_sort_simple_seq, sort_30) {
  // Create data
  std::vector<int> in = {1,  5,  7, 3,  7,  3,  8,   4,  1,   6,   4,  6, 7,   3, 12,  21,
                         65, 43, 1, 54, 34, 76, 345, 21, 765, 346, 34, 1, 434, 8, 343, 88};
  std::vector<int> out(in.size(), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  solovyev_d_shell_sort_simple_seq::TaskSequential task_sequential(task_data_seq);
  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();
  ASSERT_TRUE(solovyev_d_shell_sort_simple_seq::isSorted(out));
}