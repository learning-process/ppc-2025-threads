#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "../include/integrator.hpp"

using namespace khasanyanov_k_trapezoid_method_seq;

TEST(khasanyanov_k_trapezoid_method_seq, test_integrator_linear_function) {
  // (5x + 2y - 3z)dxdydz;
  auto f = [](const std::vector<double>& x) -> double { return (5 * x[0]) + (2 * x[1]) - (3 * x[2]); };

  std::vector<std::pair<double, double>> bounds = {{-3, 1.0}, {0.0, 2.0}, {0.5, 1.0}};

  double precision = 1e-6;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(-21.0, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_seq, test_integrator_quad_function) {
  // (x^2 + 2y - 6.5z^2)dxdydz;
  auto f = [](const std::vector<double>& x) -> double { return (x[0] * x[0]) + (2 * x[1]) - (6.5 * x[2] * x[2]); };

  std::vector<std::pair<double, double>> bounds = {{-3, 1.0}, {0.0, 2.0}, {0.5, 1.0}};

  double precision = 0.01;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(2.16666, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_seq, test_integrator_mixed_function) {
  // (x^2 + 2xy + z)dxdydz;
  auto f = [](const std::vector<double>& x) -> double { return (x[0] * x[0]) + (2 * x[1] * x[0]) + x[2]; };

  std::vector<std::pair<double, double>> bounds = {{-2.5, 0.0}, {0.0, 3.0}, {2.0, 2.5}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(2.1875, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_seq, test_integrator_trigonometric_function) {
  // (5x + 2y - 3z)dxdydz;
  auto f = [](const std::vector<double>& x) -> double { return (sin(x[0]))-x[1]; };

  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 2.0}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(-1.08060, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_seq, test_integrator_long_function) {
  // (5x + 2y - 3z)dxdydz;
  auto f = [](const std::vector<double>& x) -> double {
    return x[0] + (x[1] / 2.0) - (x[2] / 3.0) + (x[3] / 4.0) - (x[4] / 5.0);
  };

  std::vector<std::pair<double, double>> bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(0.60833, result, precision);
}