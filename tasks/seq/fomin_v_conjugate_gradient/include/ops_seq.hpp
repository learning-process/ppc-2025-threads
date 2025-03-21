#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_conjugate_gradient {

class fomin_v_conjugate_gradient_seq : public ppc::core::Task {
 public:
  explicit fomin_v_conjugate_gradient_seq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double DotProduct(const std::vector<double>& a, const std::vector<double>& b);
  std::vector<double> MatrixVectorMultiply(const std::vector<double>& A, const std::vector<double>& x);
  std::vector<double> VectorAdd(const std::vector<double>& a, const std::vector<double>& b);
  std::vector<double> VectorSub(const std::vector<double>& a, const std::vector<double>& b);
  std::vector<double> VectorScalarMultiply(const std::vector<double>& v, double scalar);

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> output_;
  int n_;
};

}  // namespace fomin_v_conjugate_gradient