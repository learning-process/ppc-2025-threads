// Copyright 2025 Kalinin Dmitry
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_jarvis_convex_hull_tbb {

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
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> points_;
  std::vector<Point> convexHullPoints_;
  size_t hull_size_ = 0;
};

std::vector<Point> Jarvis(const std::vector<Point>& points);
Point FindNextPoint(const Point& prev_point, const std::vector<Point>& points);
Point FindStartingPoint(const std::vector<Point>& points);

}  // namespace kalinin_d_jarvis_convex_hull_tbb
