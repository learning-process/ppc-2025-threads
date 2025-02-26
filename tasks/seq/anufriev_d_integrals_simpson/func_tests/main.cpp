#include <gtest/gtest.h>

#define USE_MATH_DEFINES
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/anufriev_d_integrals_simpson/include/ops_seq.hpp"

namespace {
std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& elements, size_t out_size = 1) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<double> out_buffer(out_size, 0.0);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(elements.data())),
                                 elements.size() * sizeof(double));
  task_data->inputs_count.push_back(static_cast<std::uint32_t>(elements.size()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_buffer.data()), out_buffer.size() * sizeof(double));
  task_data->outputs_count.push_back(static_cast<std::uint32_t>(out_size));

  return task_data;
}
double GetResultFromTaskData(const std::shared_ptr<ppc::core::TaskData>& td) {
  auto res_ptr = reinterpret_cast<double*>(td->outputs[0]);
  return res_ptr[0];
}
}  // namespace

TEST(anufriev_d_integrals_simpson_seq, test_1D_sin) {
  std::vector<double> in = {1, 0.0, M_PI / 2.0, 100, 1};

  auto td = MakeTaskData(in, 1);
  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  double result = GetResultFromTaskData(td);
  EXPECT_NEAR(result, 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_seq, test_2D_sum_of_squares) {
  std::vector<double> in = {2, 0.0, 1.0, 100, 0.0, 1.0, 100, 0};

  auto td = MakeTaskData(in, 1);
  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  double result = GetResultFromTaskData(td);
  EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_seq, test_2D_sin_cos) {
  std::vector<double> in = {2, 0.0, M_PI / 2.0, 100, 0.0, M_PI / 2.0, 100, 1};

  auto td = MakeTaskData(in, 1);
  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  double result = GetResultFromTaskData(td);
  EXPECT_NEAR(result, 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_seq, test_unknown_func) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 999};
  auto td = MakeTaskData(in, 1);
  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  double val = GetResultFromTaskData(td);
  EXPECT_DOUBLE_EQ(val, 0.0);
}

TEST(anufriev_d_integrals_simpson_seq, test_invalid_empty_input) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(nullptr);
  td->inputs_count.push_back(0);
  td->outputs.push_back(nullptr);
  td->outputs_count.push_back(0);

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_seq, test_invalid_dimension_zero) {
  std::vector<double> in = {0, 0.0, 1.0, 2, 999};
  auto td = MakeTaskData(in, 1);

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_seq, test_invalid_not_enough_data) {
  std::vector<double> in = {
      2, 0.0, 1.0, 2.0, 999.0,
  };
  auto td = MakeTaskData(in, 1);

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_seq, test_invalid_odd_n) {
  std::vector<double> in = {1, 0.0, 1.0, 3, 0};
  auto td = MakeTaskData(in, 1);
  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);

  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_seq, test_invalid_negative_n) {
  std::vector<double> in = {1, 0.0, 1.0, -2, 0};
  auto td = MakeTaskData(in, 1);
  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);

  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_seq, test_no_output_buffer) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size()));

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(td);
  EXPECT_FALSE(task.Validation());
}