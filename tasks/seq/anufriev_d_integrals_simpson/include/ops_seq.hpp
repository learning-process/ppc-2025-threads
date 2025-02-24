#pragma once

#include <utility>

#include "core/task/include/task.hpp"

namespace anufriev_d_integrals_simpson_seq {

class IntegralsSimpsonSequential : public ppc::core::Task {
 public:
  explicit IntegralsSimpsonSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double ax_{}, bx_{};
  double ay_{}, by_{};
  int nx_{}, ny_{};
  int func_code_{};
  double result_{};

  [[nodiscard]]  double Function(double x, double y) const;
};

}  // namespace anufriev_d_integrals_simpson_seq