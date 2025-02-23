#pragma once

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace chizhov_m_trapezoid_method_seq {
using Function = std::function<double(const std::vector<double>&)>;

double Trapezoid_method(const Function f, int div, int dim, std::vector<double>& lower_limits,
                        std::vector<double>& upper_limits);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Function f;
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
  int div;
  int dim;
  double res;
};
}  // namespace chizhov_m_trapezoid_method_seq