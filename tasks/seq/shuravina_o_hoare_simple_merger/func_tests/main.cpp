#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shuravina_o_hoare_simple_merger/include/ops_seq.hpp"

TEST(shuravina_o_hoare_simple_merger, test_sort_and_merge) {
  std::vector<int> in = {5, 2, 9, 1, 5, 6};
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = {1, 2, 5, 5, 6, 9};
  EXPECT_EQ(out, expected);
}