#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shulpin_i_jarvis_seq {

struct Point {
  double x, y;
  Point() : x(0), y(0) {}
  Point(double x_coordinate, double y_coordinate) : x(x_coordinate), y(y_coordinate) {}
};

struct CircleParams {
  double radius;
  size_t num_points;

  CircleParams() : radius(0), num_points(0) {}
  CircleParams(double new_radius, size_t new_num_points) : radius(new_radius), num_points(new_num_points) {}
};

void TestBody(std::vector<shulpin_i_jarvis_seq::Point>& input, std::vector<shulpin_i_jarvis_seq::Point>& expected);

class JarvisSequential : public ppc::core::Task {
 public:
  explicit JarvisSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static int Orientation(const Point& p, const Point& q, const Point& r);
  static void MakeJarvisPassage(std::vector<shulpin_i_jarvis_seq::Point>& input,
                                std::vector<shulpin_i_jarvis_seq::Point>& output);

 private:
  std::vector<shulpin_i_jarvis_seq::Point> input_, output_;
};

}  // namespace shulpin_i_jarvis_seq