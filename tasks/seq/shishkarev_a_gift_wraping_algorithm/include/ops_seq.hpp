#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gift_wraping_algorithm_seq {

struct Vertex {
  int x, y;
  bool operator<(const Vertex& other) const {
    if (x != other.x) { return {x < other.x}; }
    return {y < other.y};
  }
  bool operator==(const Vertex& v) const { return (x == v.x) && (y == v.y); }
  bool operator!=(const Vertex& v) const { return (x != v.x) || (y != v.y); }
  [[nodiscard]] int Length(const Vertex& v) const { return ((x - v.x) * (x - v.x)) + ((y - v.y) * (y - v.y)); }
  [[nodiscard]] double Angle(const Vertex& v, const Vertex& w) const {
    return ((v.x - x) * (w.y - y)) - ((v.y - y) * (w.x - x));
  }
};

std::vector<Vertex> RemoveDuplicates(const std::vector<Vertex>& points);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Vertex> input_, output_;
  int rc_size_{};
};

}  // namespace shishkarev_a_gift_wraping_algorithm_seq