// Copyright 2025 Kalinin Dmitry
#include "seq/kalinin_d_jarvis_convex_hull/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<Point> Jarvis(const std::vector<Point>& Points) {
  if (Points.size() < 3) return Points;

  Point p0 = Points[0];
  for (const auto& p : Points) {
    if (p.x < p0.x || (p.x == p0.x && p.y < p0.y)) p0 = p;
  }
  std::vector<Point> convexHull = {p0};
  Point prevPoint = p0;

  while (true) {
    Point nextPoint = Points[0];
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
  internal_order_test();
  // Init value for input and output
  points.resize(taskData->inputs_count[0]);
  auto* tmp_ptr_A = reinterpret_cast<Point*>(taskData->inputs[0]);
  std::copy_n(tmp_ptr_A, taskData->inputs_count[0], points.begin());
  return true;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::ValidationImpl() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return false;
  }

  std::sort(points.begin(), points.end());
  return std::unique(points.begin(), points.end()) == points.end();
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::RunImpl() {
  internal_order_test();
  convexHullPoints = Jarvis(points);
  return true;
}

bool kalinin_d_jarvis_convex_hull_seq::TestTaskSequential::PostProcessingImpl() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<Point*>(taskData->outputs[0]);
  std::copy_n(convexHullPoints.begin(), convexHullPoints.size(), output_ptr);
  return true;
}