#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shkurinskaya_e_bin_labeling/include/ops_seq.hpp"

TEST(shkurinskaya_e_bin_labeling, empty_input) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  // Create data
  std::vector<int> in;
  std::vector<int> out(size);
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling::TestTaskSequential testTaskSequential(task_data_seq);
  ASSERT_EQ(testTaskSequential.Validation(), false);
}

TEST(shkurinskaya_e_bin_labeling, empty_output) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out;
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling::TestTaskSequential testTaskSequential(task_data_seq);
  ASSERT_EQ(testTaskSequential.Validation(), false);
}

TEST(shkurinskaya_e_bin_labeling, test_diag_object) {
  int height = 100;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int i = 0; i < height; ++i) {
    in[(i * width) + i] = 1;
    ans[(i * width) + i] = 1;
  }
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling::TestTaskSequential testTaskSequential(task_data_seq);
  ASSERT_EQ(testTaskSequential.Validation(), true);
  testTaskSequential.PreProcessing();
  testTaskSequential.Run();
  testTaskSequential.PostProcessing();
  ASSERT_EQ(ans, out);
}

TEST(shkurinskaya_e_bin_labeling, test_two_components) {
  int height = 100;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  in[0] = 1;
  in[9999] = 1;
  ans[0] = 1;
  ans[9999] = 2;
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling::TestTaskSequential testTaskSequential(task_data_seq);
  ASSERT_EQ(testTaskSequential.Validation(), true);
  testTaskSequential.PreProcessing();
  testTaskSequential.Run();
  testTaskSequential.PostProcessing();
  ASSERT_EQ(ans, out);
}
