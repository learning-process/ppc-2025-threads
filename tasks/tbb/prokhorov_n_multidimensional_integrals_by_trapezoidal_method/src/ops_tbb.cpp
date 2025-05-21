#include "tbb/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb {

bool TestTaskTBB::PreProcessingImpl() {
  auto* lower_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

  dimensions_ = static_cast<int>(task_data->inputs_count[0] / sizeof(double));
  if (dimensions_ == 0) {
    return false;
  }

  lower_limits_ = std::vector<double>(lower_ptr, lower_ptr + dimensions_);
  upper_limits_ = std::vector<double>(upper_ptr, upper_ptr + dimensions_);
  steps_ = std::vector<int>(steps_ptr, steps_ptr + dimensions_);

  result_ = 0.0;
  return true;
}

bool TestTaskTBB::ValidationImpl() {
  if (task_data->inputs_count[0] == 0 || task_data->inputs_count[1] == 0 || task_data->inputs_count[2] == 0) {
    return false;
  }
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

namespace {

class RecursiveIntegrator {
 public:
  RecursiveIntegrator(const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& lower,
                      const std::vector<double>& upper, const std::vector<int>& steps)
      : func_(func), lower_(lower), upper_(upper), steps_(steps) {}

  double Compute() {
    std::vector<double> point(lower_.size());
    std::vector<double> h(lower_.size());
    for (size_t i = 0; i < lower_.size(); ++i) {
      h[i] = (upper_[i] - lower_[i]) / steps_[i];
    }
    return ParallelTrapezoidalIntegration(0, point, h);
  }

 private:
  double ParallelTrapezoidalIntegration(size_t current_dim, std::vector<double>& point, const std::vector<double>& h) {
    if (current_dim == lower_.size()) {
      return func_(point);
    }

    double sum = 0.0;

    if (current_dim == 0) {
      sum = tbb::parallel_reduce(
          tbb::blocked_range<int>(0, steps_[current_dim] + 1), 0.0,
          [&](const tbb::blocked_range<int>& r, double local_sum) {
            for (int i = r.begin(); i < r.end(); ++i) {
              point[current_dim] = lower_[current_dim] + i * h[current_dim];
              double weight = (i == 0 || i == steps_[current_dim]) ? 0.5 : 1.0;
              local_sum += weight * SequentialTrapezoidalIntegration(current_dim + 1, point, h);
            }
            return local_sum;
          },
          std::plus<>());
    } else {
      for (int i = 0; i <= steps_[current_dim]; ++i) {
        point[current_dim] = lower_[current_dim] + i * h[current_dim];
        double weight = (i == 0 || i == steps_[current_dim]) ? 0.5 : 1.0;
        sum += weight * SequentialTrapezoidalIntegration(current_dim + 1, point, h);
      }
    }

    return sum * h[current_dim];
  }

  double SequentialTrapezoidalIntegration(size_t current_dim, std::vector<double>& point,
                                          const std::vector<double>& h) {
    if (current_dim == lower_.size()) {
      return func_(point);
    }

    double sum = 0.0;
    for (int i = 0; i <= steps_[current_dim]; ++i) {
      point[current_dim] = lower_[current_dim] + i * h[current_dim];
      double weight = (i == 0 || i == steps_[current_dim]) ? 0.5 : 1.0;
      sum += weight * SequentialTrapezoidalIntegration(current_dim + 1, point, h);
    }

    return sum * h[current_dim];
  }

  const std::function<double(const std::vector<double>&)>& func_;
  const std::vector<double>& lower_;
  const std::vector<double>& upper_;
  const std::vector<int>& steps_;
};

}  // namespace

bool TestTaskTBB::RunImpl() {
  if (!function_) {
    return false;
  }

  RecursiveIntegrator integrator(function_, lower_limits_, upper_limits_, steps_);
  result_ = integrator.Compute();

  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_tbb