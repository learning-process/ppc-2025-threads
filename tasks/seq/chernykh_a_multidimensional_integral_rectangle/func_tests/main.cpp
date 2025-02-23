#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/chernykh_a_multidimensional_integral_rectangle/include/ops_seq.hpp"

namespace {

using namespace chernykh_a_multidimensional_integral_rectangle_seq;

void RunValidTask(const Func& func, BoundsPerDim& bounds_per_dim, StepsPerDim& steps_per_dim, const double want,
                  const double tolerance) {
  double output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds_per_dim.data()));
  task_data->inputs_count.emplace_back(bounds_per_dim.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps_per_dim.data()));
  task_data->inputs_count.emplace_back(steps_per_dim.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = SequentialTask(task_data, func);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(want, output, tolerance);
}

void RunInvalidTask(const Func& func, BoundsPerDim& bounds_per_dim, StepsPerDim& steps_per_dim) {
  double output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(bounds_per_dim.data()));
  task_data->inputs_count.emplace_back(bounds_per_dim.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps_per_dim.data()));
  task_data->inputs_count.emplace_back(steps_per_dim.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = SequentialTask(task_data, func);
  ASSERT_FALSE(task.Validation());
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, quadratic_3d_integration) {
  auto func = [](const Point& point) -> double {
    return (point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2]);
  };
  auto bounds_per_dim = BoundsPerDim{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps_per_dim = StepsPerDim{150, 150, 150};
  double want = 1.0;
  double tolerance = 10e-2;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, exponential_3d_integration) {
  auto func = [](const Point& point) -> double { return std::exp(point[0] + point[1] + point[2]); };
  auto bounds_per_dim = BoundsPerDim{{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}};
  auto steps_per_dim = StepsPerDim{150, 150, 150};
  double want = std::pow(std::exp(0.5) - 1, 3);
  double tolerance = 10e-3;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, trigonometric_2d_integration) {
  auto func = [](const Point& point) -> double { return std::sin(point[0]) * std::cos(point[1]); };
  auto bounds_per_dim = BoundsPerDim{{0.0, std::numbers::pi}, {0.0, std::numbers::pi / 2}};
  auto steps_per_dim = StepsPerDim{2000, 2000};
  double want = 2.0;
  double tolerance = 10e-4;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, polynomial_3d_integration) {
  auto func = [](const Point& point) -> double {
    return (point[0] * point[1]) + (point[1] * point[2]) + (point[0] * point[2]);
  };
  auto bounds_per_dim = BoundsPerDim{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps_per_dim = StepsPerDim{150, 150, 150};
  double want = 0.75;
  double tolerance = 10e-2;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, quadratic_2d_integration) {
  auto func = [](const Point& point) -> double { return (point[0] * point[0]) + (point[1] * point[1]); };
  auto bounds_per_dim = BoundsPerDim{{0.0, 2.0}, {0.0, 3.0}};
  auto steps_per_dim = StepsPerDim{2000, 2000};
  double want = 26.0;
  double tolerance = 10e-2;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, exponential_2d_integration) {
  auto func = [](const Point& point) -> double { return std::exp(point[0] + point[1]); };
  auto bounds_per_dim = BoundsPerDim{{0.0, 1.0}, {0.0, 1.0}};
  auto steps_per_dim = StepsPerDim{2000, 2000};
  double want = std::pow(std::numbers::e - 1, 2);
  double tolerance = 10e-3;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, trigonometric_3d_integration) {
  auto func = [](const Point& point) -> double { return std::sin(point[0]) * std::cos(point[1]) * std::tan(point[2]); };
  auto bounds_per_dim = BoundsPerDim{
      {0.0, std::numbers::pi},
      {0.0, std::numbers::pi / 2},
      {0.0, std::numbers::pi / 4},
  };
  auto steps_per_dim = StepsPerDim{150, 150, 150};
  double want = std::numbers::ln2;
  double tolerance = 10e-3;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, cubic_2d_integration) {
  auto func = [](const Point& point) -> double {
    return (point[0] * point[0] * point[0]) + (point[1] * point[1] * point[1]);
  };
  auto bounds_per_dim = BoundsPerDim{{0.0, 1.0}, {0.0, 2.0}};
  auto steps_per_dim = StepsPerDim{2000, 2000};
  double want = 4.5;
  double tolerance = 10e-3;
  RunValidTask(func, bounds_per_dim, steps_per_dim, want, tolerance);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, empty_bounds_fails_validation) {
  auto func = [](const Point& point) -> double { return (2 * point[0]) + (3 * point[1]); };
  auto bounds_per_dim = BoundsPerDim{};
  auto steps_per_dim = StepsPerDim{2000, 2000};
  RunInvalidTask(func, bounds_per_dim, steps_per_dim);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, incorrect_bounds_fails_validation) {
  auto func = [](const Point& point) -> double { return std::exp(point[0] + point[1] + point[2]); };
  auto bounds_per_dim = BoundsPerDim{{0.0, 0.5}, {1.0, 0.5}, {0.0, 0.5}};
  auto steps_per_dim = StepsPerDim{150, 150, 150};
  RunInvalidTask(func, bounds_per_dim, steps_per_dim);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, empty_steps_fails_validation) {
  auto func = [](const Point& point) -> double {
    return (point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2]);
  };
  auto bounds_per_dim = BoundsPerDim{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps_per_dim = StepsPerDim{};
  RunInvalidTask(func, bounds_per_dim, steps_per_dim);
}

TEST(chernykh_a_multidimensional_integral_rectangle_seq, bounds_steps_size_mismatch_fails_validation) {
  auto func = [](const Point& point) -> double {
    return (point[0] * point[1]) + (point[1] * point[2]) + (point[0] * point[2]);
  };
  auto bounds_per_dim = BoundsPerDim{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  auto steps_per_dim = StepsPerDim{2000, 2000};
  RunInvalidTask(func, bounds_per_dim, steps_per_dim);
}

}  // namespace
