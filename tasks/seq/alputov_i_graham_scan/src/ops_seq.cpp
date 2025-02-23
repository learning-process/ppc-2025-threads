#include "seq/alputov_i_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstddef>

namespace alputov_i_graham_scan_seq {

bool TestTaskSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count[0] <= task_data->outputs_count[0] && task_data->inputs_count[0] >= 3);
}

double TestTaskSequential::Cross(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

Point TestTaskSequential::FindPivot() const {
  return *std::ranges::min_element(input_points_);
}

std::vector<Point> TestTaskSequential::SortPoints(const Point& pivot) const {
  std::vector<Point> points = input_points_;
  points.erase(std::ranges::remove(points, pivot).begin(), points.end());
  std::ranges::sort(points);
  auto last = std::ranges::unique(points);
  points.erase(last.begin(), points.end());

  std::ranges::sort(points, [&pivot](const Point& a, const Point& b) {
    const double angle_a = atan2(a.y - pivot.y, a.x - pivot.x);
    const double angle_b = atan2(b.y - pivot.y, b.x - pivot.x);
    return (angle_a < angle_b) || (angle_a == angle_b && a.x < b.x);
  });
  return points;
}

std::vector<Point> TestTaskSequential::BuildHull(const std::vector<Point>& sorted_points) const {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size());
  hull.push_back(FindPivot());
  hull.push_back(sorted_points[0]);
  hull.push_back(sorted_points[1]);

  for (size_t i = 2; i < sorted_points.size(); ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }
  return hull;
}

bool TestTaskSequential::RunImpl() {
  if (input_points_.size() < 3) {
    return false;
  }

  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);
  convex_hull_ = BuildHull(sorted_points);

  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(convex_hull_, output_ptr);
  return true;
}

}  // namespace alputov_i_graham_scan_seq#include "seq/alputov_i_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstddef>

namespace alputov_i_graham_scan_seq {

bool TestTaskSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count[0] <= task_data->outputs_count[0] && task_data->inputs_count[0] >= 3);
}

double TestTaskSequential::Cross(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

Point TestTaskSequential::FindPivot() const {
  return *std::ranges::min_element(input_points_);
}

std::vector<Point> TestTaskSequential::SortPoints(const Point& pivot) const {
  std::vector<Point> points = input_points_;
  points.erase(std::ranges::remove(points, pivot).begin(), points.end());
  std::ranges::sort(points);
  auto last = std::ranges::unique(points);
  points.erase(last.begin(), points.end());

  std::ranges::sort(points, [&pivot](const Point& a, const Point& b) {
    const double angle_a = atan2(a.y - pivot.y, a.x - pivot.x);
    const double angle_b = atan2(b.y - pivot.y, b.x - pivot.x);
    return (angle_a < angle_b) || (angle_a == angle_b && a.x < b.x);
  });
  return points;
}

std::vector<Point> TestTaskSequential::BuildHull(const std::vector<Point>& sorted_points) const {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size());
  hull.push_back(FindPivot());
  hull.push_back(sorted_points[0]);
  hull.push_back(sorted_points[1]);

  for (size_t i = 2; i < sorted_points.size(); ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }
  return hull;
}

bool TestTaskSequential::RunImpl() {
  if (input_points_.size() < 3) {
    return false;
  }

  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);
  convex_hull_ = BuildHull(sorted_points);

  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(convex_hull_, output_ptr);
  return true;
}

}  // namespace alputov_i_graham_scan_seq