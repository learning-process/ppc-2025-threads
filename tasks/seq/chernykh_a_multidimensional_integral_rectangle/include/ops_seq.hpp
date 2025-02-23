#pragma once

#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_rectangle_seq {

using Point = std::vector<double>;
using Func = std::function<double(const Point&)>;
using Bounds = std::pair<double, double>;
using BoundsPerDim = std::vector<Bounds>;
using Steps = int;
using StepsPerDim = std::vector<int>;

class SequentialTask final : public ppc::core::Task {
 public:
  explicit SequentialTask(ppc::core::TaskDataPtr task_data, Func func)
      : Task(std::move(task_data)), func_(std::move(func)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Func func_;
  BoundsPerDim bounds_per_dim_;
  StepsPerDim steps_per_dim_;
  double result_{};

  [[nodiscard]] std::vector<double> GetStepSizePerDim() const;
  [[nodiscard]] int GetTotalPoints() const;
  [[nodiscard]] Point GetPoint(int idx, const std::vector<double>& step_size_per_dim) const;
  [[nodiscard]] double GetScalingFactor(const std::vector<double>& step_size_per_dim) const;
};

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
