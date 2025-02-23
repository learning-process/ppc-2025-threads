#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace poroshin_v_multi_integral_with_trapez_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> &task_data,
                              std::function<double(std::vector<double> &args)> func)
      : Task(task_data), dim(task_data->inputs_count[0]), func(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static double area(std::vector<double> &arguments);
  static double f1(std::vector<double> &arguments);
  static double f1cos(std::vector<double> &arguments);
  static double f1Euler(std::vector<double> &arguments);
  static double f3(std::vector<double> &arguments);
  static double f3advanced(std::vector<double> &arguments);

 private:
  void count_multi_integral_trapez_method_seq();
  std::vector<std::pair<double, double>> limits;
  size_t dim;
  std::function<double(std::vector<double> &args)> func;
  std::vector<int> n;
  double res{};
};

}  // namespace poroshin_v_multi_integral_with_trapez_method_seq