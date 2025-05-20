#include "omp/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>  
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp {
namespace {

double ParallelTrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                                      const std::vector<double>& lower, const std::vector<double>& upper,
                                      const std::vector<int>& steps, size_t current_dim, std::vector<double>& point) {
  if (current_dim == lower.size()) {
    return func(point);
  }

  double h = (upper[current_dim] - lower[current_dim]) / steps[current_dim];
  double sum = 0.0;

  if (point.size() <= current_dim) {
    point.resize(current_dim + 1);
  }

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i <= steps[current_dim]; ++i) {
    point[current_dim] = lower[current_dim] + i * h;
    double weight = (i == 0 || i == steps[current_dim]) ? 0.5 : 1.0;
    sum += weight * ParallelTrapezoidalIntegration(func, lower, upper, steps, current_dim + 1, point);
  }

  return sum * h;
}

}  // namespace

TestTaskOpenMP::TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool TestTaskOpenMP::PreProcessingImpl() {
  auto* lower_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

  dimensions_ = static_cast<int>(task_data->inputs_count[0] / sizeof(double));

  lower_limits_ = std::vector<double>(lower_ptr, lower_ptr + dimensions_);
  upper_limits_ = std::vector<double>(upper_ptr, upper_ptr + dimensions_);
  steps_ = std::vector<int>(steps_ptr, steps_ptr + dimensions_);

  result_ = 0.0;
  return true;
}

bool TestTaskOpenMP::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

bool TestTaskOpenMP::RunImpl() {
  std::vector<double> point;

#pragma omp parallel
  {
#pragma omp single
    result_ = ParallelTrapezoidalIntegration(function_, lower_limits_, upper_limits_, steps_, 0, point);
  }

  return true;
}

bool TestTaskOpenMP::PostProcessingImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  if (output_ptr == nullptr) {
    return false;
  }
  *output_ptr = result_;
  return true;
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_omp