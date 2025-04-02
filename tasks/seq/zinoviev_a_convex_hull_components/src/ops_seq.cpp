#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace zinoviev_a_convex_hull_components_seq;

bool ConvexHullSequential::PreProcessingImpl() {
  auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  int width = task_data->inputs_count[0];
  int height = task_data->inputs_count[1];

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (input_data[y * width + x] != 0) {
        input_points_.push_back({x, y});
      }
    }
  }
  return true;
}

bool ConvexHullSequential::ValidationImpl() {
  if (task_data->inputs_count.size() != 2) return false;
  if (task_data->outputs_count.size() != 1) return false;
  return true;
}

int ConvexHullSequential::cross(const Point& O, const Point& A, const Point& B) {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

std::vector<Point> ConvexHullSequential::findConvexHull(const std::vector<Point>& points) {
  int n = points.size();
  if (n <= 1) return points;

  std::vector<Point> hull;
  std::vector<Point> sorted_points = points;
  std::sort(sorted_points.begin(), sorted_points.end());

  for (int i = 0; i < n; ++i) {
    while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }

  int lower_hull_size = hull.size() + 1;
  for (int i = n - 2; i >= 0; --i) {
    while (hull.size() >= lower_hull_size && cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }

  hull.pop_back();
  return hull;
}

bool ConvexHullSequential::RunImpl() {
  output_hull_ = findConvexHull(input_points_);
  return true;
}

bool ConvexHullSequential::PostProcessingImpl() {
  auto* output_data = reinterpret_cast<Point*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_hull_.size(); ++i) {
    output_data[i] = output_hull_[i];
  }
  task_data->outputs_count[0] = output_hull_.size();
  return true;
}