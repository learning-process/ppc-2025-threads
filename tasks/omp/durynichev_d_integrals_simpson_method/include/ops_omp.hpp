#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace durynichev_d_integrals_simpson_method_omp {
class SimpsonIntegralOpenMP : public ppc::core::Task {
 public:
  explicit SimpsonIntegralOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> boundaries_;
  double result_;
  int n_;
  int dim_;

  double func1D(double x);
  double func2D(double x, double y);
  double simpson1D(double a, double b);
  double simpson2D(double x0, double x1, double y0, double y1);
};
}  // namespace durynichev_d_integrals_simpson_method_omp