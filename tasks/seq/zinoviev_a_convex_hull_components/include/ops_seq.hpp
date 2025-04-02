#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_convex_hull_components_seq {

struct Point {
  int x, y;
  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
  bool operator<(const Point& other) const { return x < other.x || (x == other.x && y < other.y); }
};

class ConvexHullSequential : public ppc::core::Task {
 public:
  explicit ConvexHullSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> output_hull_;
  static std::vector<Point> findConvexHull(const std::vector<Point>& points);
  static int cross(const Point& O, const Point& A, const Point& B);
};

}  // namespace zinoviev_a_convex_hull_components_seq