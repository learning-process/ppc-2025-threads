#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "seq/deryabin_m_hoare_sort_simple_merge/include/ops_seq.hpp"

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_short_array) {
  // Create data
  double input_array[6] = {-1, -2, -3, -11, -22, -33};
  size_t chunk_count = 2;
  double output_array[6]{};
  std::vector<double*> out_array(1, output_array);
  double true_solution[6] = {-33, -22, -11, -3, -2, -1};

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_array));
  task_data_seq->inputs_count.emplace_back(sizeof(input_array) / sizeof(input_array[0]));
  task_data_seq->inputs_count.emplace_back(reinterpret_cast<uint8_t*>(chunk_count));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(sizeof(output_array) / sizeof(output_array[0]));

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}
