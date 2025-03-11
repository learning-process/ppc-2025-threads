#pragma once

#include <cmath>
#include <functional>
#include <memory>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_seq {

class Integral : public ppc::core::Task {
 public:
  explicit Integral(std::shared_ptr<ppc::core::TaskData> task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void SetFunction(const std::function<double(double)>& func);

 private:
  double down_limit_;
  double up_limit_;
  int count_;
  double result_;
  std::function<double(double)> func_;
  std::shared_ptr<ppc::core::TaskData> task_data_;
  static double Compute(const std::function<double(double)>& f, double a, double b, int n);
};

}  // namespace shurigin_s_integrals_square_seq
