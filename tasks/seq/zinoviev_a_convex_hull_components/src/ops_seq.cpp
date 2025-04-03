#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace zinoviev_a_convex_hull_components_seq;

bool ConvexHullSequential::PreProcessingImpl() noexcept {
  auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int width = task_data->inputs_count[0];
  const int height = task_data->inputs_count[1];

  input_points_.clear();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (input_data[y * width + x] != 0) {
        input_points_.emplace_back(Point{x, y});
      }
    }
  }
  return true;
}

bool ConvexHullSequential::ValidationImpl() noexcept {
  return task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 1;
}

int ConvexHullSequential::cross(const Point& O, const Point& A, const Point& B) noexcept {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

std::vector<Point> ConvexHullSequential::findConvexHull(const std::vector<Point>& points) noexcept {
  const size_t n = points.size();
  if (n <= 1) return points;

  std::vector<Point> hull;
  std::vector<Point> sorted_points = points;
  std::sort(sorted_points.begin(), sorted_points.end());

  for (size_t i = 0; i < n; ++i) {
    while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }

  const size_t lower_hull_size = hull.size() + 1;
  for (size_t i = n; i-- > 0;) {
    while (hull.size() >= lower_hull_size && cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }

  hull.pop_back();
  return hull;
}

bool ConvexHullSequential::RunImpl() noexcept {
  output_hull_ = findConvexHull(input_points_);
  return true;
}

bool ConvexHullSequential::PostProcessingImpl() noexcept {
  auto* output_data = reinterpret_cast<Point*>(task_data->outputs[0]);
  const size_t hull_size = output_hull_.size();

  for (size_t i = 0; i < hull_size; ++i) {
    output_data[i] = output_hull_[i];
  }
  task_data->outputs_count[0] = static_cast<int>(hull_size);
  return true;
}