#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_simpson_seq {

typedef double (*func)(double x);

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

  func f_;
};
}  // namespace filateva_e_simpson_seq