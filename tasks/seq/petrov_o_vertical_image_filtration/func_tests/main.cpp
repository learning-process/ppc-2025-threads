#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/petrov_o_vertical_image_filtration/include/ops_seq.hpp"

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_3x3) {
  constexpr size_t kWidth = 3;
  constexpr size_t kHeight = 3;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Check result
  std::vector<int> expected_out = {5};
  EXPECT_EQ(out, expected_out);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_5x5) {
  constexpr size_t kWidth = 5;
  constexpr size_t kHeight = 5;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Check result
  std::vector<int> expected_out = {7, 8, 9, 12, 13, 14, 17, 18, 19};
  EXPECT_EQ(out, expected_out);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_const_data) {
  constexpr size_t kWidth = 5;
  constexpr size_t kHeight = 5;

  // Create data
  std::vector<int> in(kWidth * kHeight, 3);
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);
  std::vector<int> expected_out((kWidth - 2) * (kHeight - 2), 3);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Check result
  EXPECT_EQ(out, expected_out);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_negative_data) {
  constexpr size_t kWidth = 5;
  constexpr size_t kHeight = 5;

  // Create data
  std::vector<int> in = {-1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10, -11, -12, -13,
                         -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Check result
  std::vector<int> expected_out = {-7, -8, -9, -12, -13, -14, -17, -18, -19};
  EXPECT_EQ(out, expected_out);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_large_data) {
  constexpr size_t kWidth = 500;
  constexpr size_t kHeight = 500;

  // Create data
  std::vector<int> in(kWidth * kHeight, 1);
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);
  std::vector<int> expected_out((kWidth - 2) * (kHeight - 2), 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Check result
  EXPECT_EQ(out, expected_out);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_rectangular) {
  constexpr size_t kWidth = 4;
  constexpr size_t kHeight = 3;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> out((kWidth - 2) * (kHeight - 2), 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  // Check result
  std::vector<int> expected_out = {6, 7};
  EXPECT_EQ(out, expected_out);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_1x1) {
  constexpr size_t kWidth = 1;
  constexpr size_t kHeight = 1;

  // Create data
  std::vector<int> in = {1};
  std::vector<int> out(0, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(petrov_o_vertical_image_filtration_seq, test_gaussian_filter_empty) {
  constexpr size_t kWidth = 0;
  constexpr size_t kHeight = 0;

  // Create data
  std::vector<int> in;
  std::vector<int> out;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(kWidth);
  task_data_seq->inputs_count.emplace_back(kHeight);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  petrov_o_vertical_image_filtration_seq::TaskSequential test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.Validation(), false);
}