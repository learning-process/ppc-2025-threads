#include "seq/zinoviev_a_convex_hull_components/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"

using namespace zinoviev_a_convex_hull_components_seq;

ConvexHullSequential::ConvexHullSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ConvexHullSequential::PreProcessingImpl() noexcept {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  const auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int width = static_cast<int>(task_data->inputs_count[0]);
  const int height = static_cast<int>(task_data->inputs_count[1]);
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
         task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

int ConvexHullSequential::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

std::vector<Point> ConvexHullSequential::FindConvexHull(const std::vector<Point>& points) noexcept {
  const size_t n = points.size();
  if (n < 3) return points;

  std::vector<Point> sorted_points(points);
  std::sort(sorted_points.begin(), sorted_points.end());

  std::vector<Point> hull;
  hull.reserve(2 * n);

  for (const auto& point : sorted_points) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), point) <= 0) {
      hull.pop_back();
    }
    hull.push_back(point);
  }

  hull.pop_back();
  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (!hull.empty()) hull.pop_back();
  return hull;
}

bool ConvexHullSequential::RunImpl() noexcept {
  output_hull_ = FindConvexHull(input_points_);
  return true;
}

bool ConvexHullSequential::PostProcessingImpl() noexcept {
  if (task_data->outputs.empty() || static_cast<size_t>(task_data->outputs_count[0]) < output_hull_.size()) {
    return false;
  }

  auto* output = reinterpret_cast<Point*>(task_data->outputs[0]);
  const size_t n = output_hull_.size();

  for (size_t i = 0; i < n; ++i) {
    output[i] = output_hull_[i];
  }

  task_data->outputs_count[0] = static_cast<int>(n);
  return true;
}