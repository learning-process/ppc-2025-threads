#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vedernikova_k_gauss_seq/include/ops_seq.hpp"

using TaskVars = std::tuple<uint32_t, uint32_t, uint32_t, Image, Image>;

namespace {
class vedernikova_k_gauss_test_seq  // NOLINT(readability-identifier-naming)
    : public ::testing::TestWithParam<TaskVars> {
 protected:
};

TEST_F(vedernikova_k_gauss_test_seq, validation_fails_not_enough_params) {
  Image in(15, 128);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vedernikova_k_gauss_seq::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_F(vedernikova_k_gauss_test_seq, validation_fails_no_input_image) {
  Image in(15, 128);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  vedernikova_k_gauss_seq::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_F(vedernikova_k_gauss_test_seq, validation_fails_no_output_buffer) {
  Image in(15, 128);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(1);
  task_data->outputs_count.emplace_back(out.size());

  vedernikova_k_gauss_seq::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_F(vedernikova_k_gauss_test_seq, validation_fails_in_and_out_sizes_are_different) {
  Image in(15, 128);
  Image out(in.size());
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(5);
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() + 1);

  vedernikova_k_gauss_seq::Gauss task(task_data);

  EXPECT_FALSE(task.Validation());
}

TEST_P(vedernikova_k_gauss_test_seq, returns_correct_blurred_image) {
  const auto &[width, height, channels, in, exp] = GetParam();
  Image out(in.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs_count.emplace_back(channels);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  vedernikova_k_gauss_seq::Gauss task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, exp);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(vedernikova_k_gauss_test_seq, vedernikova_k_gauss_test_seq, ::testing::Values(
    TaskVars(
      1, 1, 1,
      {255},
      {255}
    ),
    TaskVars(
      3, 3, 1, 
      {
        15, 15, 15,
        15, 15, 15,
        15, 15, 15,
      },
      {
        15, 15, 15,
        15, 15, 15,
        15, 15, 15,
      }
    ),
    TaskVars(
      5, 5, 1, 
      {
        255, 0, 255, 0, 255, 
        0, 255, 0, 255, 0, 
        255, 0, 255, 0, 255, 
        0, 255, 0, 255, 0, 
        255, 0, 255, 0, 255, 
      }, 
      {
        231, 37, 219, 37, 231,
        37, 209, 47, 209, 37,
        219, 47, 209, 47, 219,
        37, 209, 47, 209, 37,
        231, 37, 219, 37, 231,
      }
    ),
    TaskVars(
      1000, 1000, 1,
      Image(1000000, (uint8_t)128),
      Image(1000000, (uint8_t)128)
    )
    
  )
);
}  // namespace