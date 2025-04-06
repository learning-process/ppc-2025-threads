#ifndef TASKS_OMP_DURYNICHEV_D_INTEGRALS_SIMPSON_METHOD_INCLUDE_OPS_OMP_HPP_
#define TASKS_OMP_DURYNICHEV_D_INTEGRALS_SIMPSON_METHOD_INCLUDE_OPS_OMP_HPP_

#include <cstddef>  // Для size_t
#include <memory>   // Для std::shared_ptr и std::move
#include <utility>  // Для std::move
#include <vector>   // Для std::vector

#include "core/task/include/task.hpp"

namespace durynichev_d_integrals_simpson_method_omp {

class SimpsonIntegralOpenMP : public ppc::core::Task {
 public:
  explicit SimpsonIntegralOpenMP(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> boundaries_;
  int n_{};
  size_t dim_{};
  double result_{};

  static double Func1D(double x);
  static double Func2D(double x, double y);
  double Simpson1D(double a, double b);
  double Simpson2D(double x0, double x1, double y0, double y1);
};

}  // namespace durynichev_d_integrals_simpson_method_omp

#endif  // TASKS_OMP_DURYNICHEV_D_INTEGRALS_SIMPSON_METHOD_INCLUDE_OPS_OMP_HPP_