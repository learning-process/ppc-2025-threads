// Copyright 2025 Kalinin Dmitry
#include "omp/kalinin_d_jarvis_convex_hull/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

std::vector<kalinin_d_jarvis_convex_hull_omp::Point> kalinin_d_jarvis_convex_hull_omp::Jarvis(
    const std::vector<kalinin_d_jarvis_convex_hull_omp::Point>& points) {
  if (points.size() < 3) {
    return points;
  }

  kalinin_d_jarvis_convex_hull_omp::Point p0 = points[0];
  for (const auto& p : points) {
    if (p.x < p0.x || (p.x == p0.x && p.y < p0.y)) {
      p0 = p;
    }
  }
  std::vector<kalinin_d_jarvis_convex_hull_omp::Point> convex_hull = {p0};
  kalinin_d_jarvis_convex_hull_omp::Point prev_point = p0;

  while (true) {
    kalinin_d_jarvis_convex_hull_omp::Point next_point = points[0];

#pragma omp parallel
    {
      kalinin_d_jarvis_convex_hull_omp::Point local_next_point = next_point;

#pragma omp for nowait
      for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const auto& point = points[i];
        if (point == prev_point) {
          continue;
        }

        double cross_product = ((point.y - prev_point.y) * (local_next_point.x - prev_point.x)) -
                               ((point.x - prev_point.x) * (local_next_point.y - prev_point.y));

        if (cross_product > 0 ||
            (cross_product == 0 && ((point.x - prev_point.x) * (point.x - prev_point.x) +
                                    (point.y - prev_point.y) * (point.y - prev_point.y)) >
                                       ((local_next_point.x - prev_point.x) * (local_next_point.x - prev_point.x) +
                                        (local_next_point.y - prev_point.y) * (local_next_point.y - prev_point.y)))) {
          local_next_point = point;
        }
      }

#pragma omp critical
      {
        double cross_product = ((local_next_point.y - prev_point.y) * (next_point.x - prev_point.x)) -
                               ((local_next_point.x - prev_point.x) * (next_point.y - prev_point.y));

        if (cross_product > 0 ||
            (cross_product == 0 && ((local_next_point.x - prev_point.x) * (local_next_point.x - prev_point.x) +
                                    (local_next_point.y - prev_point.y) * (local_next_point.y - prev_point.y)) >
                                       ((next_point.x - prev_point.x) * (next_point.x - prev_point.x) +
                                        (next_point.y - prev_point.y) * (next_point.y - prev_point.y)))) {
          next_point = local_next_point;
        }
      }
    }

    if (next_point == p0) {
      break;
    }
    convex_hull.push_back(next_point);
    prev_point = next_point;
  }

  return convex_hull;
}

bool kalinin_d_jarvis_convex_hull_omp::TestTaskOmp::PreProcessingImpl() {
  points_.resize(task_data->inputs_count[0]);
  auto* tmp_ptr_a = reinterpret_cast<kalinin_d_jarvis_convex_hull_omp::Point*>(task_data->inputs[0]);
  std::copy_n(tmp_ptr_a, task_data->inputs_count[0], points_.begin());
  return true;
}

bool kalinin_d_jarvis_convex_hull_omp::TestTaskOmp::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  std::ranges::sort(points_, [](const Point& a, const Point& b) { return a < b; });
  return std::ranges::unique(points_).begin() == points_.end();
}

bool kalinin_d_jarvis_convex_hull_omp::TestTaskOmp::RunImpl() {
  convexHullPoints_ = kalinin_d_jarvis_convex_hull_omp::Jarvis(points_);
  return true;
}

bool kalinin_d_jarvis_convex_hull_omp::TestTaskOmp::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<kalinin_d_jarvis_convex_hull_omp::Point*>(task_data->outputs[0]);
  std::copy_n(convexHullPoints_.begin(), convexHullPoints_.size(), output_ptr);
  return true;
}