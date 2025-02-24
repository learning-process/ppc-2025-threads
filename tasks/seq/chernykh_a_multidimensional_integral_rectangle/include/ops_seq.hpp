#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_rectangle_seq {

using Point = std::vector<double>;
using Function = std::function<double(const Point &)>;

struct Dimension {
  double lower_bound_{};
  double upper_bound_{};
  int steps_count_{};

  bool IsValid() const;
  double GetStepSize() const;
};

class SequentialTask final : public ppc::core::Task {
 public:
  explicit SequentialTask(ppc::core::TaskDataPtr task_data, Function func)
      : Task(std::move(task_data)), func_(std::move(func)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Function func_;
  std::vector<Dimension> dims_;
  double result_{};

  int GetTotalPoints() const;
  Point GetPoint(int index) const;
  double GetScalingFactor() const;
};

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
