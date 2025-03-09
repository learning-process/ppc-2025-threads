// Copyright 2025 Kalinin Dmitry
#include "seq/kalinin_d_jarvis_convex_hull/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<kalinin_d_jarvis_convex_hull_seq::Point> kalinin_d_jarvis_convex_hull_seq::Jarvis(
    const std::vector<kalinin_d_jarvis_convex_hull_seq::Point>& Points) {
  if (Points.size() < 3) return Points;

  kalinin_d_jarvis_convex_hull_seq::Point p0 = Points[0];
  for (const auto& p : Points) {
    if (p.x < p0.x || (p.x == p0.x && p.y < p0.y)) p0 = p;
  }
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> convexHull = {p0};
  kalinin_d_jarvis_convex_hull_seq::Point prevPoint = p0;

  while (true) {
    kalinin_d_jarvis_convex_hull_seq::Point nextPoint = Points[0];
    for (const auto& point : Points) {
      if (point == prevPoint) continue;

      double crossProduct =
          (point.y - prevPoint.y) * (nextPoint.x - prevPoint.x) - (point.x - prevPoint.x) * (nextPoint.y - prevPoint.y);

      if (crossProduct > 0 || (crossProduct == 0 && ((point.x - prevPoint.x) * (point.x - prevPoint.x) +
                                                     (point.y - prevPoint.y) * (point.y - prevPoint.y)) >
                                                        ((nextPoint.x - prevPoint.x) * (nextPoint.x - prevPoint.x) +
                                                         (nextPoint.y - prevPoint.y) * (nextPoint.y - prevPoint.y)))) {
        nextPoint = point;
      }
    }

    if (nextPoint == p0) break;
    convexHull.push_back(nextPoint);
    prevPoint = nextPoint;
  }

  return convexHull;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  points.resize(task_data->inputs_count[0]);
  auto* tmp_ptr_A = reinterpret_cast<kalinin_d_jarvis_convex_hull_seq::Point*>(task_data->inputs[0]);
  std::copy_n(tmp_ptr_A, task_data->inputs_count[0], points.begin());
  return true;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  std::sort(points.begin(), points.end());
  return std::unique(points.begin(), points.end()) == points.end();
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::RunImpl() {
  convexHullPoints = Jarvis(points);
  return true;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<kalinin_d_jarvis_convex_hull_seq::Point*>(task_data->outputs[0]);
  std::copy_n(convexHullPoints.begin(), convexHullPoints.size(), output_ptr);
  return true;
}