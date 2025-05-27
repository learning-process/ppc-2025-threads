#pragma once

#include <utility>
#include <vector>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace fomin_v_conjugate_gradient {

class FominVConjugateGradientAll : public ppc::core::Task {
 public:
  explicit FominVConjugateGradientAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  FominVConjugateGradientAll() : world_(MPI_COMM_WORLD) {}

  static double DotProduct(const std::vector<double>& a, const std::vector<double>& b);
  [[nodiscard]] std::vector<double> MatrixVectorMultiply(const std::vector<double>& a,
                                                         const std::vector<double>& x) const;
  static std::vector<double> VectorAdd(const std::vector<double>& a, const std::vector<double>& b);
  static std::vector<double> VectorSub(const std::vector<double>& a, const std::vector<double>& b);
  static std::vector<double> VectorScalarMultiply(const std::vector<double>& v, double scalar);

  int n;

 private:
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> output_;
  boost::mpi::communicator world_;
  std::vector<double> local_a_;  
  std::vector<double> local_b_;
};

}  // namespace fomin_v_conjugate_gradient