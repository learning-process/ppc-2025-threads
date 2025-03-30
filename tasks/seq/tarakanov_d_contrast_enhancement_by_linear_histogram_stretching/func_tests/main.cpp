#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_contrast_enhancement_by_linear_histogram_stretching/include/ops_seq.hpp"

TEST(tarakanov_d_linear_stretching_seq, test_contrast_stretching_random_5x5) {
  constexpr size_t kSize = 5;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  std::vector<unsigned char> in(kSize * kSize);
  for (size_t i = 0; i < kSize * kSize; ++i) {
    in[i] = static_cast<unsigned char>(dis(gen));
  }

  std::vector<unsigned char> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  tarakanov_d_linear_stretching::TaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.Validation());

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  unsigned char min_out = 255;
  unsigned char max_out = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    min_out = std::min(min_out, out[i]);
    max_out = std::max(max_out, out[i]);
  }

  EXPECT_EQ(min_out, 0);
  EXPECT_EQ(max_out, 255);
}

TEST(tarakanov_d_linear_stretching_seq, test_contrast_stretching_big_image_random) {
  constexpr size_t kSize = 5;

  std::random_device rd;
  std::mt19937 gen(rd());
  // std::uniform_int_distribution<> dis(0, 127);
  std::uniform_int_distribution<> dis(0, 255);

  std::vector<unsigned char> in(kSize * kSize);
  for (size_t i = 0; i < kSize * kSize; ++i) {
    in[i] = static_cast<unsigned char>(dis(gen));
  }

  std::vector<unsigned char> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  tarakanov_d_linear_stretching::TaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.Validation());

  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  unsigned char min_out = 255;
  unsigned char max_out = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    min_out = std::min(min_out, out[i]);
    max_out = std::max(max_out, out[i]);
  }

  EXPECT_EQ(min_out, 0);
  EXPECT_EQ(max_out, 255);
}
