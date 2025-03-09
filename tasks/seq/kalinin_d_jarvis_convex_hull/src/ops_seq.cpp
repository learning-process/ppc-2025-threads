// Copyright 2025 Kalinin Dmitry
#include "seq/kalinin_d_jarvis_convex_hull/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace std::chrono_literals;

std::vector<kalinin_d_jarvis_convex_hull_seq::Point> kalinin_d_jarvis_convex_hull_seq::Jarvis(
    const std::vector<kalinin_d_jarvis_convex_hull_seq::Point>& points) {
  if (points.size() < 3) {
    return points;
  }

  kalinin_d_jarvis_convex_hull_seq::Point p0 = points[0];
  for (const auto& p : points) {
    if (p.x < p0.x || (p.x == p0.x && p.y < p0.y)) {
      p0 = p
    };
  }
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> convex_hull = {p0};
  kalinin_d_jarvis_convex_hull_seq::Point prev_point = p0;

  while (true) {
    kalinin_d_jarvis_convex_hull_seq::Point next_point = points[0];
    for (const auto& point : points) {
      if (point == prev_point) {
        continue;
      }

      double cross_product = ((point.y - prev_point.y) * (next_point.x - prev_point.x)) -
                             ((point.x - prev_point.x) * (next_point.y - prev_point.y));

      if (cross_product > 0 ||
          (cross_product == 0 &&
           ((point.x - prev_point.x) * (point.x - prev_point.x) + (point.y - prev_point.y) * (point.y - prev_point.y)) >
               ((next_point.x - prev_point.x) * (next_point.x - prev_point.x) +
                (next_point.y - prev_point.y) * (next_point.y - prev_point.y)))) {
        next_point = point;
      }
    }

    if (next_point == p0) {
      break;
    }
    convexHull.push_back(next_point);
    prev_point = next_point;
  }

  return convexHull;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  points.resize(task_data->inputs_count[0]);
  auto* tmp_ptr_a = reinterpret_cast<kalinin_d_jarvis_convex_hull_seq::Point*>(task_data->inputs[0]);
  std::copy_n(tmp_ptr_a, task_data->inputs_count[0], points.begin());
  return true;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  std::ranges::sort(points.begin(), points.end());
  return std::ranges::unique(points.begin(), points.end()) == points.end();
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::RunImpl() {
  convexHullPoints = kalinin_d_jarvis_convex_hull_seq::Jarvis(points);
  return true;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<kalinin_d_jarvis_convex_hull_seq::Point*>(task_data->outputs[0]);
  std::copy_n(convexHullPoints.begin(), convexHullPoints.size(), output_ptr);
  return true;
}