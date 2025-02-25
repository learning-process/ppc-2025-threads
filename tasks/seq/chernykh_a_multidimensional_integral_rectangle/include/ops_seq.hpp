#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_rectangle_seq {

using Point = std::vector<double>;
using Function = std::function<double(const Point &)>;

struct Dimension {
  Dimension(const double lower_bound, const double upper_bound, const int steps_count)
      : lower_bound(lower_bound), upper_bound(upper_bound), steps_count(steps_count) {}
  double lower_bound{};
  double upper_bound{};
  int steps_count{};

  [[nodiscard]] bool IsValid() const;
  [[nodiscard]] double GetStepSize() const;
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

  [[nodiscard]] int GetTotalPoints() const;
  [[nodiscard]] Point GetPoint(int index) const;
  [[nodiscard]] double GetScalingFactor() const;
};

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
