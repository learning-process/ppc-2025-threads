#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zolotareva_a_SLE_gradient_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  inline static void ConjugateGradient(const std::vector<double>& A, const std::vector<double>& b,
                                       std::vector<double>& x, int N);
  inline static void DotProduct(double& sum, const std::vector<double>& vec1, const std::vector<double>& vec2, int n);
  inline static void MatrixVectorMult(const std::vector<double>& matrix, const std::vector<double>& vector,
                                      std::vector<double>& result, int n);
  inline static bool IsPositiveAndSimm(const double* A, int n);

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  int n_{0};
};

}  // namespace zolotareva_a_SLE_gradient_method_seq