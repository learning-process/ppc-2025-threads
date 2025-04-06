#include <cstddef>  // Для size_t
#include <omp.h>
#include <vector>

#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

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

#ifdef PERF_TEST
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1D(double x) {
  double result = 0.0;
  for (int i = 0; i < 1000; ++i) {
    double t = x + (i * 0.001);
    result += std::exp(-t * t) * std::sin(t) * std::cos(t * 2.0) / (std::log(t + 1.1) + 1.0);
  }
  return result;
}

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2D(double x, double y) {
  double result = 0.0;
  for (int i = 0; i < 1000; ++i) {
    double tx = x + (i * 0.001);
    double ty = y + (i * 0.001);
    result += std::exp(-tx * tx - ty * ty) * std::sin(tx * ty) * std::cos(tx + ty) / (std::log(tx + ty + 1.1) + 1.0);
  }
  return result;
}
#else
double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func1D(double x) { return (x * x); }

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Func2D(double x, double y) {
  return (x * x) + (y * y);
}
#endif

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson1D(double a, double b) const {
  double h = (b - a) / n_;
  double sum = Func1D(a) + Func1D(b);  // Исправлен синтаксис
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

double durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP::Simpson2D(double x0, double x1,
                                                                                   double y0, double y1) {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i <= n_; i++) {
    double x = x0 + (i * hx);
    double coef_x = 0.0;
    if (i == 0 || i == n_) {
      coef_x = 1;
    } else if (i % 2 != 0) {
      coef_x = 4;
    } else {
      coef_x = 2;
    }
    double local_sum = 0.0;

    for (int j = 0; j <= n_; j++) {
      double y = y0 + (j * hy);
      double coef_y = 0.0;
      if (j == 0 || j == n_) {
        coef_y = 1;
      } else if (j % 2 != 0) {
        coef_y = 4;
      } else {
        coef_y = 2;
      }
      local_sum += coef_x * coef_y * Func2D(x, y);
    }
    sum += local_sum;
  }

  return sum * hx * hy / 9.0;
}