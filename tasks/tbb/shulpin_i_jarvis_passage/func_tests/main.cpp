#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/shulpin_i_jarvis_passage/include/test_modules.hpp"

namespace {
std::vector<shulpin_i_jarvis_tbb::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_tbb::Point &center,
                                                                double radius) {
  std::vector<shulpin_i_jarvis_tbb::Point> points;
  for (size_t i = 0; i < num_points; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points);
    double x = center.x + (radius * std::cos(angle));
    double y = center.y + (radius * std::sin(angle));
    points.emplace_back(x, y);
  }
  return points;
}
}  // namespace

TEST(shulpin_i_jarvis_tbb, square_with_point) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, ox_line) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, triangle) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {3, 0}, {1, 2}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {3, 0}, {1, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, octagone) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, repeated_points) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, real_case) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, star_case) {
  // clang-format off
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0.0, 3.0},
    {1.0, 1.0},
    {3.0, 1.0},
    {1.5, -0.5},
    {2.5, -3.0},
    {0.0, -1.5},
    {-2.5, -3.0},
    {-1.5, -0.5},
    {-3.0, 1.0},
    {-1.0, 1.0},
    {0.0, 3.0}
  };
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{-3.0, 1.0},
      {-2.5, -3.0},
      {2.5, -3.0},
      {3.0, 1.0},
      {0.0, 3.0},
  };
  // clang-format on
  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_tbb, one_point_validation_false) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{0, 0}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{0, 0}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_tbb, three_points_validation_false) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {{1, 1}, {2, 2}};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {{1, 1}, {2, 2}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_tbb, zero_points_validation_false) {
  std::vector<shulpin_i_jarvis_tbb::Point> input = {};
  std::vector<shulpin_i_jarvis_tbb::Point> expected = {};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_tbb, circle_r10_p100) {
  shulpin_i_jarvis_tbb::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_tbb::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_tbb, circle_r10_p200) {
  shulpin_i_jarvis_tbb::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 200;

  std::vector<shulpin_i_jarvis_tbb::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_tbb, random_10_points) {
  size_t num_points = 10;

  std::vector<shulpin_i_jarvis_tbb::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, random_50_points) {
  size_t num_points = 50;

  std::vector<shulpin_i_jarvis_tbb::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_omp, random_100_points) {
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_tbb::Point> input = GenerateRandomPoints(num_points);
  std::vector<shulpin_i_jarvis_tbb::Point> expected = ComputeConvexHull(input);

  RandomTestBody(input, expected);
}