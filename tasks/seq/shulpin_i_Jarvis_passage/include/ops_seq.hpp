#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shulpin_i_jarvis_seq {

struct Point {
  double x, y;
  Point() : x(0), y(0) {}
  Point(double xCoordinate, double yCoordinate) : x(xCoordinate), y(yCoordinate) {}
};

class JarvisSequential : public ppc::core::Task {
 public:
  explicit JarvisSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int Orientation(const Point& p, const Point& q, const Point& r);
  void MakeJarvisPassage(std::vector<shulpin_i_jarvis_seq::Point>& input_,
                         std::vector<shulpin_i_jarvis_seq::Point>& output_);
  /*bool comparePoints(const shulpin_i_jarvis_seq::Point& a, const shulpin_i_jarvis_seq::Point& b);*/

 private:
  std::vector<shulpin_i_jarvis_seq::Point> input_, output_;
};

}  // namespace shulpin_i_jarvis_seq