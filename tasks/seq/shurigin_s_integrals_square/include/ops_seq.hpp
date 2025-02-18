#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_seq {

class Integral : public ppc::core::Task {
 public:
  explicit Integral(std::shared_ptr<ppc::core::TaskData> taskData_);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void setFunction(const std::function<double(double)>& func);

 private:
  double down_limit;
  double up_limit;
  int count;
  double result_;
  std::function<double(double)> func_;
  std::shared_ptr<ppc::core::TaskData> taskData;
  static double compute(const std::function<double(double)>& f, double a, double b, int n);
};

}  // namespace shurigin_s_integrals_square_seq
