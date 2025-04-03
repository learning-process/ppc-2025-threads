#pragma once

#include <compare>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_convex_hull_components_seq {

struct Point {
  int x, y;
  auto operator<=>(const Point&) const = default;
  bool operator==(const Point&) const = default;
};

class ConvexHullSequential : public ppc::core::Task {
 public:
  explicit ConvexHullSequential(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() noexcept override;
  bool ValidationImpl() noexcept override;
  bool RunImpl() noexcept override;
  bool PostProcessingImpl() noexcept override;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> output_hull_;

  static std::vector<Point> FindConvexHull(const std::vector<Point>& points) noexcept;
  static int Cross(const Point& o, const Point& a, const Point& b) noexcept;
};

}  // namespace zinoviev_a_convex_hull_components_seq