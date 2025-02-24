#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/laganina_e_component_labeling/include/ops_seq.hpp"

TEST(laganina_e_component_labeling_seq, validation_test1) {
  int m = 1;
  int n = -1;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
  test_task_sequential.PreProcessing();
}

TEST(laganina_e_component_labeling_seq, validation_test4) {
  int m = -1;
  int n = 1;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
  test_task_sequential.PreProcessing();
}

TEST(laganina_e_component_labeling_seq, validation_test2) {
  int m = -1;
  int n = -1;
  // Create data
  std::vector<int> in(m * n, 3);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
  test_task_sequential.PreProcessing();
}

TEST(laganina_e_component_labeling_seq, validation_test3) {
  int m = 3;
  int n = -1;
  // Create data
  std::vector<int> in(m * n, 3);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
  test_task_sequential.PreProcessing();
}

TEST(laganina_e_component_labeling_seq, all_one) {
  int m = 3;
  int n = 2;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(laganina_e_component_labeling_seq, all_zero) {
  int m = 3;
  int n = 2;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}

TEST(laganina_e_component_labeling_seq, test1) {
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, test2) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, test3) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, test6) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, test7) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, one_row) {
  int m = 1;
  int n = 6;
  // Create data
  std::vector<int> in = {1, 0, 1, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 2, 0, 3};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, test4) {
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(laganina_e_component_labeling_seq, test5) {
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  // Create Task
  laganina_e_component_labeling_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}