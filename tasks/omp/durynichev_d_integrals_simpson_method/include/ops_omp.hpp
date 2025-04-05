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
#ifdef PERF_TEST
  double func1D(double x);  // Версия с нагрузкой для perf_tests
  double func2D(double x, double y);
#else
  double func1D(double x);  // Оригинальная версия для func_tests
  double func2D(double x, double y);
#endif
  double simpson1D(double a, double b);
  double simpson2D(double x0, double x1, double y0, double y1);

  std::vector<double> boundaries_;
  double result_;
  int n_;
  int dim_;
};
}  // namespace durynichev_d_integrals_simpson_method_omp