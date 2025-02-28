#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/morozov_e_lineare_image_filtering_block_gaussian/include/ops_seq.hpp"

TEST(morozov_e_lineare_image_filtering_block_gaussian, empty_image_test) {
  int n = 0;
  int m = 0;
  std::vector<std::vector<int>> image(n, std::vector<int>(m));
  std::vector<std::vector<int>> image_res(n, std::vector<int>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, size_input_not_equal_size_output_test1) {
  int n = 0;
  int m = 0;
  std::vector<std::vector<int>> image(n, std::vector<int>(m));
  std::vector<std::vector<int>> image_res(n + 1, std::vector<int>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, size_input_not_equal_size_output_test2) {
  int n = 0;
  int m = 0;
  std::vector<std::vector<int>> image(n, std::vector<int>(m));
  std::vector<std::vector<int>> image_res(n, std::vector<int>(m + 1));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, main_test1) {
  int n = 5;
  int m = 5;
  std::vector<std::vector<double>> image = {
      {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  std::vector image_res(n, std::vector<double>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(image, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, main_test2) {
  int n = 5;
  int m = 5;
  std::vector<std::vector<double>> image = {
      {2, 2, 3, 2, 2}, {2, 2, 3, 2, 2}, {2, 2, 3, 2, 2}, {2, 2, 3, 2, 2}, {2, 2, 3, 2, 2}};
  std::vector image_res(n, std::vector<double>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<std::vector<double>> real_res = {
      {2, 2, 3, 2, 2}, {2, 2.25, 2.5, 2.25, 2}, {2, 2.25, 2.5, 2.25, 2}, {2, 2.25, 2.5, 2.25, 2}, {2, 2, 3, 2, 2}};
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, main_test3) {
  int n = 5;
  int m = 5;
  std::vector<std::vector<double>> image = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
  std::vector image_res(n, std::vector<double>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<std::vector<double>> real_res = {
      {1, 2, 3, 4, 5}, {6, 4.5, 5.5, 6.5, 10}, {1, 3.25, 4.25, 5.25, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, main_test4) {
  int n = 5;
  int m = 5;
  std::vector<std::vector<double>> image = {
      {5, 5, 5, 5, 5}, {5, 5, 5, 5, 5}, {5, 5, 5, 5, 5}, {5, 5, 5, 5, 5}, {5, 5, 5, 5, 5}};
  std::vector image_res(n, std::vector<double>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(image, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, main_test5) {
  int n = 5;
  int m = 5;
  std::vector<std::vector<double>> image = {
      {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {10, 9, 8, 7, 6}, {5, 4, 3, 2, 1}, {1, 1, 1, 1, 1}};
  std::vector image_res(n, std::vector<double>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<std::vector<double>> real_res = {
      {1, 2, 3, 4, 5}, {6, 6.25, 6.75, 7.25, 10}, {10, 7.25, 6.75, 6.25, 6}, {5, 4.5, 3.75, 3, 1}, {1, 1, 1, 1, 1}};
  EXPECT_EQ(real_res, image_res);
}
TEST(morozov_e_lineare_image_filtering_block_gaussian, main_test6) {
  int n = 3;
  int m = 3;
  std::vector<std::vector<double>> image = {{1, 6, 7}, {8, 2, 1}, {8, 2, 4}};
  std::vector image_res(n, std::vector<double>(m));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<std::vector<double>> real_res = {{1, 6, 7}, {8, 3.875, 1}, {8, 2, 4}};
  EXPECT_EQ(real_res, image_res);
}
