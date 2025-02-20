#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace oturin_a_gift_wrapping_seq {

struct coord {
  int x, y;
  bool operator==(coord o) { return (x == o.x && y == o.y); }
  bool operator!=(coord o) { return !(x == o.x && y == o.y); }
};

coord randCoord(int r);

double distance(coord a, coord b);

// Angle Between Three Points
double ABTP(coord c, coord b, coord a);

// Angle Between Three Points for leftmost point
double ABTP(coord a, coord c);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<coord> input_, output_;
  int n;
};

}  // namespace oturin_a_gift_wrapping_seq