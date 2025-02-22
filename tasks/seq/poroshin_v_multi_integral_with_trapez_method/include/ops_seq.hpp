#pragma once

#include <math.h>

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace poroshin_v_multi_integral_with_trapez_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData>& taskData,
                              std::function<double(std::vector<double>& args)> func)
      : Task(taskData), dim(taskData->inputs_count[0]), func(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void count_multi_integral_trapez_method_seq();
  std::vector<std::pair<double, double>> limits;
  size_t dim;
  std::function<double(std::vector<double>& args)> func;
  std::vector<int> n;
  double res{};
};

}  // namespace poroshin_v_multi_integral_with_trapez_method_seq