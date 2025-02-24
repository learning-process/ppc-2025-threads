#include "seq/anufriev_d_integrals_simpson/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace anufriev_d_integrals_simpson_seq {

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}

double IntegralsSimpsonSequential::Function(double x, double y) const {
  switch (func_code_) {
    case 0:
      return (x * x) + (y * y);
    case 1:
      return std::sin(x) * std::cos(y);
    default:
      return 0.0;
  }
}

bool IntegralsSimpsonSequential::PreProcessingImpl() {
  if (task_data->inputs.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t in_size = task_data->inputs_count[0];

  if (in_size < 6) {
    return false;
  }

  ax_ = in_ptr[0];
  bx_ = in_ptr[1];
  nx_ = static_cast<int>(in_ptr[2]);
  ay_ = in_ptr[3];
  by_ = in_ptr[4];
  ny_ = static_cast<int>(in_ptr[5]);

  if (nx_ <= 0 || ny_ <= 0) {
    return false;
  }
  if ((nx_ % 2) != 0 || (ny_ % 2) != 0) {
    return false;
  }

  if (in_size >= 7) {
    func_code_ = static_cast<int>(in_ptr[6]);
  } else {
    func_code_ = 0;
  }

  result_ = 0.0;

  return true;
}

bool IntegralsSimpsonSequential::ValidationImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  if (task_data->outputs_count[0] < 1) {
    return false;
  }

  return true;
}

bool IntegralsSimpsonSequential::RunImpl() {
  double hx = (bx_ - ax_) / nx_;
  double hy = (by_ - ay_) / ny_;

  double sum = 0.0;
  for (int j = 0; j <= ny_; ++j) {
    double yj = ay_ + (j * hy);
    int cj = SimpsonCoeff(j, ny_);

    for (int i = 0; i <= nx_; ++i) {
      double xi = ax_ + (i * hx);
      int ci = SimpsonCoeff(i, nx_);

      sum += ci * cj * Function(xi, yj);
    }
  }

  double coeff = (hx * hy) / 9.0;
  result_ = coeff * sum;

  return true;
}

bool IntegralsSimpsonSequential::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  out_ptr[0] = result_;

  return true;
}

}  // namespace anufriev_d_integrals_simpson_seq