#include "tbb/zinoviev_a_convex_hull_components/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_convex_hull_components_tbb {

ConvexHullTBB::ConvexHullTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ConvexHullTBB::PreProcessingImpl() noexcept {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  const auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int width = static_cast<int>(task_data->inputs_count[0]);
  const int height = static_cast<int>(task_data->inputs_count[1]);

  tbb::concurrent_vector<Point> conc_points;

  tbb::parallel_for(0, height, [&](int y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      if (input_data[idx] != 0) {
        conc_points.push_back(Point{x, y});
      }
    }
  });

  input_points_.assign(conc_points.begin(), conc_points.end());
  return true;
}

bool ConvexHullTBB::ValidationImpl() noexcept {
  return task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 1 &&
         task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

int ConvexHullTBB::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

std::vector<Point> ConvexHullTBB::FindConvexHull(const std::vector<Point>& points) noexcept {
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> sorted_points(points);
  tbb::parallel_sort(sorted_points.begin(), sorted_points.end(), [](const Point& a, const Point& b) { return a < b; });

  std::vector<Point> hull;
  hull.reserve(sorted_points.size() * 2);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  size_t lower_hull_size = hull.size();
  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (hull.size() >= lower_hull_size + 1 && Cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  hull.pop_back();
  return hull;
}

bool ConvexHullTBB::RunImpl() noexcept {
  output_hull_ = FindConvexHull(input_points_);
  return true;
}

bool ConvexHullTBB::PostProcessingImpl() noexcept {
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

}  // namespace zinoviev_a_convex_hull_components_tbb