#pragma once

#include <utility>

#include "core/task/include/task.hpp"

namespace filateva_e_simpson_omp {

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
};
}  // namespace filateva_e_simpson_omp