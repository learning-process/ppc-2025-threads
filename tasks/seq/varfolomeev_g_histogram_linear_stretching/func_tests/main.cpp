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
#include "seq/varfolomeev_g_histogram_linear_stretching/include/ops_seq.hpp"

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_9) {
  // Create data
  std::vector<int> in = {100, 50, 200, 75, 150, 25, 175, 125, 225};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = {96, 32, 223, 64, 159, 0, 191, 128, 255};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_10) {
  // Create data
  std::vector<int> in = {30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = {0, 28, 57, 85, 113, 142, 170, 198, 227, 255};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_25) {
  // Create data
  std::vector<int> in = {12,  25,  88, 14,  65,  79, 64, 128, 122, 220, 138, 147, 215,
                         211, 189, 89, 167, 181, 2,  12, 34,  25,  85,  75,  77};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = {12,  27,  101, 14,  74,  90, 73, 147, 140, 255, 159, 170, 249,
                                   244, 219, 102, 193, 209, 0,  12, 37,  27,  97,  85,  88};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_stretched) {
  // Create data (already has 0 and 255 in it)
  std::vector<int> in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 255, 19, 20, 21, 22, 23};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = in;  // nothing changes

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_boundary_values) {
  // Create data
  std::vector<int> in = {0, 1, 254, 255};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = {0, 1, 254, 255};  // nothing changes
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_flat) {
  // Create data
  std::vector<int> in = {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                         128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = in;  // nothing changes

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_single_non_flat_pixel) {
  // Create data
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_single_non_flat_pixel_2) {
  // Create data
  std::vector<int> in = {36, 36, 36, 36, 36,  36, 36, 36, 36, 36, 36, 36, 36, 36,
                         36, 36, 36, 36, 128, 36, 36, 36, 36, 36, 36, 36, 36, 36};
  std::vector<int> out(in.size());
  std::vector<int> expected_out = {0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_10k_generated) {
  // Create data
  std::vector<int> in(10000);
  std::vector<int> out(10000);
  std::vector<int> expected_out(10000);

  std::mt19937 gen(25);
  std::uniform_int_distribution<> dis(0, 255);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  int min_val = *std::min_element(in.begin(), in.end());
  int max_val = *std::max_element(in.begin(), in.end());
  if (min_val == max_val) {
    std::fill(expected_out.begin(), expected_out.end(), 0);
  } else {
    for (size_t i = 0; i < in.size(); ++i) {
      expected_out[i] =
          static_cast<int>(std::round(((in[i] - min_val) / static_cast<double>(max_val - min_val)) * 255.0));
    }
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_empty) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_negative_values) {
  // Create data
  std::vector<int> in = {-10, -5, 0, 5, 10};
  std::vector<int> out(in.size());
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_exceed_max_value) {
  // Create data
  std::vector<int> in = {250, 260, 300};
  std::vector<int> out(in.size());
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}