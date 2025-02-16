#pragma once 

#include <functional>
#include <random>
#include <numeric>

#include"core/task/include/task.hpp"

namespace lopatin_i_monte_carlo_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  using IntegrandFunction = double(const std::vector<double>&);
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data, std::function<IntegrandFunction> func)
      : Task(std::move(task_data)), integrand(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> integrationBounds;
  std::function<IntegrandFunction> integrand;
  double result{};
  int iterations;
};

} // namespace lopatin_i_monte_carlo_seq