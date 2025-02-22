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
#include "seq/kalyakina_a_Shell_with_simple_merge/include/ops_seq.hpp"

namespace kalyakina_a_Shell_with_simple_merge_seq_perf_tests {
std::vector<int> CreateReverseSortedVector(unsigned int size, const int left) {
  std::vector<int> result;
  while (size--) {
    result.push_back(left + size);
  }
  return result;
}

bool IsSorted(const std::vector<int> vec) {
  if (vec.size() < 2) {
    return true;
  }
  for (unsigned int i = 1; i < vec.size(); i++) {
    if (vec[i - 1] > vec[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace kalyakina_a_Shell_with_simple_merge_seq_perf_tests

void TestOfFunction(std::vector<int>& in) {
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSequential->inputs_count.emplace_back(in.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size());

  kalyakina_a_Shell_with_simple_merge_seq::ShellSortSequential TaskSequential(taskDataSequential);

  ASSERT_EQ(TaskSequential.Validation(), true);
  TaskSequential.PreProcessing();
  TaskSequential.Run();
  TaskSequential.PostProcessing();

  ASSERT_TRUE(kalyakina_a_Shell_with_simple_merge_seq_perf_tests::IsSorted(out));
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_of_Validation1) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSequential->inputs_count.emplace_back(0);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size());

  kalyakina_a_Shell_with_simple_merge_seq::ShellSortSequential TaskSequential(taskDataSequential);

  ASSERT_EQ(TaskSequential.Validation(), false);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_of_Validation2) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSequential->inputs_count.emplace_back(in.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(0);

  kalyakina_a_Shell_with_simple_merge_seq::ShellSortSequential TaskSequential(taskDataSequential);

  ASSERT_EQ(TaskSequential.Validation(), false);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_of_Validation3) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSequential->inputs_count.emplace_back(in.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size() + 1);

  kalyakina_a_Shell_with_simple_merge_seq::ShellSortSequential TaskSequential(taskDataSequential);

  ASSERT_EQ(TaskSequential.Validation(), false);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, small_fixed_vector) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3, 2};
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, medium_fixed_vector) {
  std::vector<int> in = {2,  9,  5,  1,  4,  8,  3,  11, 34, 12, 6,  29,  13,   7,     56,     32,      88,
                         90, 94, 78, 54, 47, 37, 77, 22, 44, 55, 66, 123, 1234, 12345, 123456, 1234567, 0};
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_50) {
  std::vector<int> in = kalyakina_a_Shell_with_simple_merge_seq_perf_tests::CreateReverseSortedVector(50, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_100) {
  std::vector<int> in = kalyakina_a_Shell_with_simple_merge_seq_perf_tests::CreateReverseSortedVector(100, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_1000) {
  std::vector<int> in = kalyakina_a_Shell_with_simple_merge_seq_perf_tests::CreateReverseSortedVector(1000, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_10000) {
  std::vector<int> in = kalyakina_a_Shell_with_simple_merge_seq_perf_tests::CreateReverseSortedVector(10000, -10);
  TestOfFunction(in);
}
