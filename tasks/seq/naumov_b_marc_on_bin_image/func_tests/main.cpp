#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/naumov_b_marc_on_bin_image/include/ops_seq.hpp"

TEST(naumov_b_marc_on_bin_image_seq, Validation_1) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 2, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);
  EXPECT_FALSE(test_task_sequential.Validation());
}

TEST(naumov_b_marc_on_bin_image_seq, Validation_2) {
  int m = 3;
  int n = 3;

  std::vector<int> in;
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);
  EXPECT_FALSE(test_task_sequential.Validation());
}

TEST(naumov_b_marc_on_bin_image_seq, Validation_3) {
  int m = 0;
  int n = 0;

  std::vector<int> in;
  std::vector<int> out;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);
  EXPECT_FALSE(test_task_sequential.Validation());
}

TEST(naumov_b_marc_on_bin_image_seq, SingleComponentInCorner) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, RingShape) {
  int m = 5;
  int n = 5;

  std::vector<int> in = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, MazeStructure) {
  int m = 7;
  int n = 7;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
                         1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 2, 0,
                              2, 0, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, SimpleTest_1) {
  int m = 3;
  int n = 4;

  std::vector<int> in = {1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, SimpleTest_2) {
  int m = 4;
  int n = 4;

  std::vector<int> in = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<int> exp_out = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, SimpleTest_3) {
  int m = 5;
  int n = 5;

  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 0, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, SingleComponent) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, OnlyBackground) {
  int m = 3;
  int n = 3;

  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> exp_out = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, MultipleComponents) {
  int m = 4;
  int n = 4;

  std::vector<int> in = {1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 0, 3, 0, 4, 5, 0, 6, 0, 0, 7, 0, 8};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, SingleRow) {
  int m = 1;
  int n = 5;

  std::vector<int> in = {1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, SingleColumn) {
  int m = 5;
  int n = 1;

  std::vector<int> in = {1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3};
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, exp_out);
}

TEST(naumov_b_marc_on_bin_image_seq, large3) {
  int m = 200;
  int n = 200;

  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(m);
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(m);
  task_data_seq->outputs_count.emplace_back(n);

  naumov_b_marc_on_bin_image_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(in, out);
}
