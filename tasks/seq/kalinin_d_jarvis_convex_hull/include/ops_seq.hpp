// Copyright 2025 Kalinin Dmitry
#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_jarvis_convex_hull_seq {

struct Point {
  int x, y;

  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
  bool operator<(const Point& other) const {
    if (x != other.x) {
      return x < other.x;
    }
    return y < other.y;
  }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskDataPtr> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> points;
  std::vector<Point> convexHullPoints;
};

std::vector<Point> Jarvis(const std::vector<Point>& points);

}  // namespace kalinin_d_jarvis_convex_hull_seq
