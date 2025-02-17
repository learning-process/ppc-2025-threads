#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shulpin_i_Jarvis_seq {

struct Point {
  double x, y;
  Point() : x(0), y(0) {}
  Point(double x, double y) : x(x), y(y) {}
};

class JarvisSequential : public ppc::core::Task {
 public:
  explicit JarvisSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int orientation(const Point& p, const Point& q, const Point& r);
  void makeJarvisPassage(std::vector<shulpin_i_Jarvis_seq::Point>& input,
                         std::vector<shulpin_i_Jarvis_seq::Point>& output);

 private:
  std::vector<shulpin_i_Jarvis_seq::Point> input, output;
};

}  // namespace shulpin_i_Jarvis_seq