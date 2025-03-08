#pragma once

#include <utility>

#include "core/task/include/task.hpp"

namespace filateva_e_simpson_stl {

using Func = double (*)(double);

class Simpson : public ppc::core::Task {
 public:
  explicit Simpson(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a_{}, b_{};
  double alfa_{};
  double res_{};

  Func f_;
  double Max_z(int start, int end);
  double Res(int start, int end, double h);
};
}  // namespace filateva_e_simpson_stl