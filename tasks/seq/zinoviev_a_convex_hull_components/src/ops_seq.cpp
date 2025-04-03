#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace zinoviev_a_convex_hull_components_seq;

ConvexHullSequential::ConvexHullSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ConvexHullSequential::PreProcessingImpl() noexcept {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  const auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int width = task_data->inputs_count[0];
  const int height = task_data->inputs_count[1];
  const size_t total_pixels = static_cast<size_t>(width) * height;

  input_points_.clear();
  input_points_.reserve(total_pixels);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      if (idx < total_pixels && input_data[idx] != 0) {
        input_points_.emplace_back(Point{x, y});
      }
    }
  }
  return true;
}

bool ConvexHullSequential::ValidationImpl() noexcept {
  return task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 1 &&
         task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->outputs_count[0] >= 0;
}

int ConvexHullSequential::cross(const Point& O, const Point& A, const Point& B) noexcept {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

std::vector<Point> ConvexHullSequential::findConvexHull(const std::vector<Point>& points) noexcept {
  if (points.size() < 2) return points;

  std::vector<Point> sorted_points(points);
  std::sort(sorted_points.begin(), sorted_points.end());

  std::vector<Point> hull;
  hull.reserve(sorted_points.size() * 2);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  const size_t lower_size = hull.size();
  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (hull.size() > lower_size && cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (!hull.empty()) hull.pop_back();
  return hull;
}

bool ConvexHullSequential::RunImpl() noexcept {
  output_hull_ = findConvexHull(input_points_);
  return true;
}

bool ConvexHullSequential::PostProcessingImpl() noexcept {
  if (task_data->outputs.empty() || static_cast<size_t>(task_data->outputs_count[0]) < output_hull_.size()) {
    return false;
  }

  auto* output_data = reinterpret_cast<Point*>(task_data->outputs[0]);
  const size_t N = output_hull_.size();

  if (output_data != nullptr) {
    for (size_t i = 0; i < N; ++i) {
      output_data[i] = output_hull_[i];
    }
    task_data->outputs_count[0] = static_cast<int>(N);
  }
  return true;
}