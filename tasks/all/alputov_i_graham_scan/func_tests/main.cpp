#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <set>
#include <vector>

#include "core/task/include/task.hpp"
#include "all/alputov_i_graham_scan/include/ops_all.hpp"  // Changed from ../include

namespace {
// Helper to convert std::vector<Point> to std::vector<double> for task input
std::vector<double> PointsToDoubles(const std::vector<alputov_i_graham_scan_all::Point>& points) {
  std::vector<double> doubles;
  doubles.reserve(points.size() * 2);
  for (const auto& p : points) {
    doubles.push_back(p.x);
    doubles.push_back(p.y);
  }
  return doubles;
}

// Helper to convert std::vector<double> (hull output) to std::vector<Point>
std::vector<alputov_i_graham_scan_all::Point> DoublesToPoints(const std::vector<double>& doubles, int hull_size) {
  std::vector<alputov_i_graham_scan_all::Point> points;
  if (hull_size == 0) return points;
  points.reserve(hull_size);
  for (int i = 0; i < hull_size; ++i) {
    points.emplace_back(doubles[2 * i], doubles[2 * i + 1]);
  }
  return points;
}

void GenerateRandomData(std::vector<alputov_i_graham_scan_all::Point>& data, size_t count) {
  std::mt19937 gen(42);  // Fixed seed for reproducible tests
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  data.clear();
  data.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    data.emplace_back(dist(gen), dist(gen));
  }
}

std::vector<alputov_i_graham_scan_all::Point> GenerateStarPoints(size_t num_points_star) {
  std::vector<alputov_i_graham_scan_all::Point> input;
  input.reserve(num_points_star * 2);
  for (size_t i = 0; i < num_points_star; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points_star);
    input.emplace_back(20.0 * std::cos(angle), 20.0 * std::sin(angle));  // Outer points
    input.emplace_back(
        5.0 * std::cos(angle + (std::numbers::pi / static_cast<double>(num_points_star))),
        5.0 * std::sin(angle + (std::numbers::pi / static_cast<double>(num_points_star))));  // Inner points
  }
  return input;
}

// Validates task execution and MPI synchronization
void ExecuteAndValidateTask(alputov_i_graham_scan_all::TestTaskALL& task) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

void AssertPointsEqual(const std::vector<alputov_i_graham_scan_all::Point>& actual,
                       const std::vector<alputov_i_graham_scan_all::Point>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  // For convex hulls, the starting point might differ but the sequence should be a cyclic shift.
  // And floating point comparisons need tolerance.
  // A robust check involves sorting both (if order doesn't matter) or checking cyclic permutations.
  // For now, simple check if sizes match. Content check done by set.
  if (actual.empty() && expected.empty()) return;
  if (actual.empty() || expected.empty()) {
    FAIL() << "One hull is empty while other is not.";
    return;
  }

  std::set<alputov_i_graham_scan_all::Point> actual_set(actual.begin(), actual.end());
  std::set<alputov_i_graham_scan_all::Point> expected_set(expected.begin(), expected.end());
  ASSERT_EQ(actual_set.size(), actual.size()) << "Actual hull has duplicate points.";
  ASSERT_EQ(expected_set.size(), expected.size()) << "Expected hull has duplicate points.";

  ASSERT_EQ(actual_set.size(), expected_set.size());

  for (const auto& p_exp : expected_set) {
    bool found = false;
    for (const auto& p_act : actual_set) {
      if (p_act == p_exp) {  // Uses Point::operator==
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Expected point (" << p_exp.x << "," << p_exp.y << ") not found in actual hull.";
  }
}

}  // namespace

TEST(alputov_i_graham_scan_all, minimal_triangle_case) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {2, 0}, {1, 2}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);  // Max possible size

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  ExecuteAndValidateTask(task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 3);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    // Expected order might vary depending on pivot and tie-breaking.
    // Check if the set of points is the same.
    std::set<alputov_i_graham_scan_all::Point> expected_set(input_points.begin(), input_points.end());
    std::set<alputov_i_graham_scan_all::Point> actual_set(actual_hull.begin(), actual_hull.end());
    ASSERT_EQ(actual_set, expected_set);
  }
}

TEST(alputov_i_graham_scan_all, collinear_points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  ExecuteAndValidateTask(task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 2);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    std::vector<alputov_i_graham_scan_all::Point> expected_hull_option1 = {{0, 0}, {3, 3}};
    std::vector<alputov_i_graham_scan_all::Point> expected_hull_option2 = {{3, 3}, {0, 0}};

    bool match = (actual_hull[0] == expected_hull_option1[0] && actual_hull[1] == expected_hull_option1[1]) ||
                 (actual_hull[0] == expected_hull_option2[0] && actual_hull[1] == expected_hull_option2[1]);
    ASSERT_TRUE(match);
  }
}

TEST(alputov_i_graham_scan_all, perfect_square_case) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  ExecuteAndValidateTask(task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 4);
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    AssertPointsEqual(actual_hull, input_points);  // Order might differ, so check sets
  }
}

TEST(alputov_i_graham_scan_all, random_100_points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points;
  GenerateRandomData(input_points, 100);
  // Add known bounding box to ensure these points are in the hull
  input_points.insert(input_points.end(), {{-1500, -1500}, {1500, -1500}, {1500, 1500}, {-1500, 1500}});

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  ExecuteAndValidateTask(task);

  if (rank == 0) {
    ASSERT_GE(hull_size_actual, 4);  // At least the bounding box
    ASSERT_LE(hull_size_actual, static_cast<int>(input_points.size()));
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);

    std::set<alputov_i_graham_scan_all::Point> actual_set(actual_hull.begin(), actual_hull.end());
    ASSERT_TRUE(actual_set.count({-1500, -1500}));
    ASSERT_TRUE(actual_set.count({1500, -1500}));
    ASSERT_TRUE(actual_set.count({1500, 1500}));
    ASSERT_TRUE(actual_set.count({-1500, 1500}));
  }
}

TEST(alputov_i_graham_scan_all, duplicate_points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points(10, {2.5, 3.5});  // 10 duplicates
  input_points.insert(input_points.end(), {{0, 0}, {5, 0}, {0, 5}, {5, 5}});   // A square

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  ExecuteAndValidateTask(task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, 4);  // The square
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);
    std::vector<alputov_i_graham_scan_all::Point> expected_square = {{0, 0}, {5, 0}, {0, 5}, {5, 5}};
    AssertPointsEqual(actual_hull, expected_square);
  }
}

TEST(alputov_i_graham_scan_all, star_figure) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t num_points_star = 10;
  std::vector<alputov_i_graham_scan_all::Point> input_points = GenerateStarPoints(num_points_star);

  std::vector<double> input_doubles = PointsToDoubles(input_points);
  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(input_points.size() * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());
  }

  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  ExecuteAndValidateTask(task);

  if (rank == 0) {
    ASSERT_EQ(hull_size_actual, static_cast<int>(num_points_star));
    std::vector<alputov_i_graham_scan_all::Point> actual_hull = DoublesToPoints(output_hull_doubles, hull_size_actual);

    std::vector<alputov_i_graham_scan_all::Point> expected_outer_points;
    for (size_t i = 0; i < num_points_star; ++i) {
      double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points_star);
      expected_outer_points.emplace_back(20.0 * std::cos(angle), 20.0 * std::sin(angle));
    }
    AssertPointsEqual(actual_hull, expected_outer_points);
  }
}

TEST(alputov_i_graham_scan_all, single_point_invalid) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<alputov_i_graham_scan_all::Point> input_points = {{0, 0}};
  std::vector<double> input_doubles = PointsToDoubles(input_points);

  int hull_size_actual = 0;
  std::vector<double> output_hull_doubles(1 * 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_doubles.data()));
    task_data->inputs_count.emplace_back(input_doubles.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hull_size_actual));
    task_data->outputs_count.emplace_back(1);  // For count
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_hull_doubles.data()));
    task_data->outputs_count.emplace_back(output_hull_doubles.size());  // For points
  }
  alputov_i_graham_scan_all::TestTaskALL task(task_data);
  if (rank == 0) {
    ASSERT_FALSE(task.ValidationImpl());  // Expecting >= 3 points for validation
  } else {                                // Other ranks always pass validation
    ASSERT_TRUE(task.ValidationImpl());
  }
  // If validation passed (e.g. if we allowed <3 points), then run the rest
  // For this test, validation is expected to fail on rank 0.
}