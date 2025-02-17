#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/lopatin_i_monte_carlo/include/lopatinMonteCarloSeq.hpp"

namespace lopatin_i_monte_carlo_seq {

std::vector<double> GenerateBounds(int dimensions, double min_val = 0.0, double max_val = 1.0) {
  std::vector<double> bounds;
  for (int i = 0; i < dimensions; ++i) {
    bounds.push_back(min_val);
    bounds.push_back(max_val);
  }
  return bounds;
}
}  // namespace lopatin_i_monte_carlo_seq

TEST(lopatin_i_monte_carlo_seq, validationInvalidInputOddBoundsCount) {
  std::vector<double> bounds = {0.0, 1.0, 2.0};  // even num of bounds
  const int iterations = 1000;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());  // incorrect num of inputs
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>&) { return 1.0; });
  ASSERT_FALSE(task.Validation());
}

TEST(lopatin_i_monte_carlo_seq, validationMissingOutputData) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(2);
  const int iterations = 1000;

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(4);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>&) { return 1.0; });
  ASSERT_FALSE(task.Validation());
}

TEST(lopatin_i_monte_carlo_seq, highDimensionalIntegration) {
  const int dimensions = 7;
  const int iterations = 20000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions, -1.0, 1.0);
  const double expected = std::pow(2.0, dimensions);  // hypercube

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>&) {
    return 1.0;  // const
  });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_NEAR(result, expected, 0.02 * expected);  // error 2%
}

TEST(lopatin_i_monte_carlo_seq, 1DConstantFunction) {
  const int dimensions = 1;
  const int iterations = 100000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions, 2.0, 5.0);  // [2, 5]
  const double expected = 3.0;                                                                   // 5 - 2 = 3

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>& x) {
    return 1.0;  // f(x) = 1
  });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_NEAR(result, expected, 0.01 * expected);  // error 1%
}

TEST(lopatin_i_monte_carlo_seq, 3DExponentialFunction) {
  const int dimensions = 3;
  const int iterations = 50000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions);  // [0,1]^3
  const double expected = std::pow(std::exp(1) - 1, 3);                                // = 5.073

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);
  lopatin_i_monte_carlo_seq::TestTaskSequential task(
      task_data, [](const std::vector<double>& x) { return std::exp(x[0] + x[1] + x[2]); });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_NEAR(result, expected, 0.02 * expected);  // error 2%
}

TEST(lopatin_i_monte_carlo_seq, 2DLinearFunction) {
  const int dimensions = 2;
  const int iterations = 20000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  auto function = [](const std::vector<double>& x) {
    assert(x.size() == 2);
    return x[0] + x[1];
  };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = 1.0;
  const double tolerance = 0.01 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 1%
}

TEST(lopatin_i_monte_carlo_seq, 3DProductFunction) {
  const int dimensions = 3;
  const int iterations = 30000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data,
                                                     [](const std::vector<double>& x) { return x[0] * x[1] * x[2]; });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = 0.125;
  const double tolerance = 0.01 * expected;
  EXPECT_NEAR(result, expected, tolerance);
}

TEST(lopatin_i_monte_carlo_seq, 4DQuadraticFunction) {
  const int dimensions = 4;
  const int iterations = 50000;  // increase for 4D
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // (x1 + x2 + x3 + x4)^2
  auto function = [](const std::vector<double>& x) {
    double sum = x[0] + x[1] + x[2] + x[3];
    return sum * sum;
  };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // anal 13/3 = 4.33333
  const double expected = 13.0 / 3.0;
  const double tolerance = 0.01 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 1%
}

TEST(lopatin_i_monte_carlo_seq, 5DExponentialFunction) {
  const int dimensions = 5;
  const int iterations = 150000;  // increase for 5D
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions, -3.0, 3.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // exp(x1 + x2 + x3 + x4 + x5)
  auto function = [](const std::vector<double>& x) { return std::exp(x[0] + x[1] + x[2] + x[3] + x[4]); };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // analytical (e^3 - e^{-3})^5 = (20.0855 - 0.0498)^5 = 3.2e6
  const double single_dim_integral = std::exp(3.0) - std::exp(-3.0);  // =20.0357
  const double expected = std::pow(single_dim_integral, 5);           // =20.0357^5 = 3.28e6

  const double tolerance = 0.07 * expected;  // error 7%
  EXPECT_NEAR(result, expected, tolerance);
}

TEST(lopatin_i_monte_carlo_seq, 2DCosineFunction) {
  const int dimensions = 2;
  const int iterations = 100000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(dimensions, 0.0, std::numbers::pi / 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // cos(x + y)
  auto function = [](const std::vector<double>& x) { return std::cos(x[0] + x[1]); };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // analytical = 0
  const double expected = 0.0;
  const double tolerance = 0.01;  // error 1%

  EXPECT_NEAR(result, expected, tolerance);
}