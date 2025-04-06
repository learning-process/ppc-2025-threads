#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

#include <omp.h>

#include <cstddef>
#include <vector>

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  boundaries_ = std::vector<double>(in_ptr, in_ptr + input_size);
  n_ = static_cast<int>(boundaries_.back());
  boundaries_.pop_back();
  dim_ = static_cast<size_t>(boundaries_.size() / 2);

  result_ = 0.0;
  return true;
}

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::ValidationImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  int n = static_cast<int>(in_ptr[task_data->inputs_count[0] - 1]);
  return task_data->inputs_count[0] >= 3 && task_data->outputs_count[0] == 1 && (n % 2 == 0);
}

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::RunImpl() {
  if (dim_ == 1) {
    result_ = Simpson1D(boundaries_[0], boundaries_[1]);
  } else if (dim_ == 2) {
    result_ = Simpson2D(boundaries_[0], boundaries_[1], boundaries_[2], boundaries_[3]);
  }
  return true;
}

bool durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1D(double x) { return (x * x); }

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2D(double x, double y) {
  return (x * x) + (y * y);
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson1D(double a, double b) const {
  double h = (b - a) / n_;
  double sum = Func1D(a) + Func1D(b);
  double sum_odd = 0.0;
  double sum_even = 0.0;

#pragma omp parallel
  {
#pragma omp for reduction(+ : sum_odd)
    for (int i = 1; i < n_; i += 2) {
      sum_odd += Func1D(a + (i * h));
    }

#pragma omp for reduction(+ : sum_even)
    for (int i = 2; i < n_ - 1; i += 2) {
      sum_even += Func1D(a + (i * h));
    }
  }

  sum += 4 * sum_odd + 2 * sum_even;
  return sum * h / 3.0;
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson2D(double x0, double x1, double y0,
                                                                                   double y1) {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i <= n_; i++) {
    double x = x0 + (i * hx);
    double coef_x = (i == 0 || i == n_) ? 1.0 : (i % 2 != 0 ? 4.0 : 2.0);
    double local_sum = 0.0;

    for (int j = 0; j <= n_; j++) {
      double y = y0 + (j * hy);
      double coef_y = (j == 0 || j == n_) ? 1.0 : (j % 2 != 0 ? 4.0 : 2.0);
      local_sum += coef_y * Func2D(x, y);
    }
    sum += coef_x * local_sum; // coef_x применяется здесь
  }

  return sum * hx * hy / 9.0;
}