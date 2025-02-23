#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_graham_scan_seq {

struct Point {
  double x, y;
  Point(double x = 0, double y = 0) : x(x), y(y) {}
  bool operator<(const Point& other) const { return std::tie(y, x) < std::tie(other.y, other.x); }
  bool operator==(const Point& other) const { return std::tie(x, y) == std::tie(other.x, other.y); }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] const std::vector<Point>& GetConvexHull() const { return convex_hull_; }

 private:
  std::vector<Point> input_points_;
  std::vector<Point> convex_hull_;

  [[nodiscard]] Point FindPivot() const;
  [[nodiscard]] std::vector<Point> SortPoints(const Point& pivot) const;
  [[nodiscard]] std::vector<Point> BuildHull(const std::vector<Point>& sorted_points) const;

  static double Cross(const Point& o, const Point& a, const Point& b);
};

}  // namespace alputov_i_graham_scan_seq