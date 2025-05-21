#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_convex_hull_components_all {

struct Point {
  int x, y;

  bool operator<(const Point& other) const { return x < other.x || (x == other.x && y < other.y); }
  bool operator==(const Point& other) const = default;
};

class ConvexHullMPI : public ppc::core::Task {
 public:
  explicit ConvexHullMPI(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() noexcept override;
  bool ValidationImpl() noexcept override;
  bool RunImpl() noexcept override;
  bool PostProcessingImpl() noexcept override;

 private:
  int rank_;
  int comm_size_;
  std::vector<std::vector<Point>> local_components_;
  std::vector<std::vector<Point>> global_components_;
  std::vector<std::vector<Point>> local_hulls_;
  std::vector<std::vector<Point>> global_hulls_;

  static std::vector<Point> FindConvexHull(const std::vector<Point>& points) noexcept;
  static int Cross(const Point& o, const Point& a, const Point& b) noexcept;
  static void BFS(const int* local_input_data, int local_width, int local_height, int start_x, int start_y,
                  std::vector<bool>& visited, std::vector<Point>& component) noexcept;
  [[nodiscard]] std::vector<int> CalculateLocalRanges(int global_size) const noexcept;
};

}  // namespace zinoviev_a_convex_hull_components_all