#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace ermolaev_v_graham_scan_seq {

class Point {
 public:
  int x;
  int y;

  Point(int x_, int y_) : x(x_), y(y_) {}
  Point() : x(0), y(0) {}
  bool operator<=(const Point& rhs) const { return y < rhs.y || (y == rhs.y && x <= rhs.x); }
  bool operator==(const Point& rhs) const { return y == rhs.y && x == rhs.x; }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_, output_;
};

}  // namespace ermolaev_v_graham_scan_seq