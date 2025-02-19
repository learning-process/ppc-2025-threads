#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_integral_simpson_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data, std::function<double(std::vector<double>)> func_)
      : Task(std::move(task_data)), func(func_) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<double> findFunctionValue(const std::vector<std::vector<double>>& coordinates,
                                        std::function<double(std::vector<double>)> f);
  void generatePointsAndEvaluate(const std::vector<std::vector<double>>& coordinates, int index,
                                 std::vector<double>& current, std::vector<double>& results,
                                 const std::function<double(const std::vector<double>)> f);
  std::vector<double> findCoeff(int count_step);
  void multiplyCoeffandFunctionValue(std::vector<double>& function_val, const std::vector<double>& coeff_vec, int a);
  double createOutputResult(std::vector<double> vec, std::vector<double> size_steps);
  bool checkBorders(std::vector<int> vec);

 private:
  double result_output = 0;
  int nums_variables = 0;
  std::vector<int> steps;
  std::vector<int> borders;
  std::function<double(std::vector<double>)> func;
};

}  // namespace kolokolova_d_integral_simpson_method_seq