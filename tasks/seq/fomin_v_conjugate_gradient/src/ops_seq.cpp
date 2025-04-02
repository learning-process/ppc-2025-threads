#include "seq/fomin_v_conjugate_gradient/include/ops_seq.hpp"

#include <cmath>
#include <vector>

double fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::DotProduct(const std::vector<double>& a,
                                                                              const std::vector<double>& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::MatrixVectorMultiply(
    const std::vector<double>& A, const std::vector<double>& x) {
  std::vector<double> result(n_, 0.0);
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      result[i] += A[i * n_ + j] * x[j];
    }
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::VectorAdd(
    const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::VectorSub(
    const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i] * scalar;
  }
  return result;
}

bool fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> input(in_ptr, in_ptr + input_size);

  n_ = static_cast<int>((-1.0 + std::sqrt(1 + 4 * input_size)) / 2);
  A_ = std::vector<double>(input.begin(), input.begin() + n_ * n_);
  b_ = std::vector<double>(input.begin() + n_ * n_, input.end());
  output_.resize(n_, 0.0);

  return true;
}

bool fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  int n = static_cast<int>((-1.0 + std::sqrt(1 + 4 * input_size)) / 2);
  if (static_cast<unsigned int>(n * (n + 1)) != input_size ||
      task_data->outputs_count[0] != static_cast<unsigned int>(n)) {
    return false;
  }
  return true;
}

bool fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::RunImpl() {
  const double epsilon = 1e-6;
  const int max_iter = 1000;  
  std::vector<double> x(n_, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rs_old = DotProduct(r, r);

  for (int i = 0; i < max_iter; ++i) {
    std::vector<double> Ap = MatrixVectorMultiply(A_, p);
    double pAp = DotProduct(p, Ap);

    if (std::abs(pAp) < 1e-12) {
      break;
    }

    double alpha = rs_old / pAp;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    std::vector<double> r_new = VectorSub(r, VectorScalarMultiply(Ap, alpha));

    double rs_new = DotProduct(r_new, r_new);
    if (std::sqrt(rs_new) < epsilon) {
      break;
    }

    double beta = rs_new / rs_old;
    p = VectorAdd(r_new, VectorScalarMultiply(p, beta));
    r = r_new;
    rs_old = rs_new;
  }

  output_ = x;
  return true;
}

bool fomin_v_conjugate_gradient::fomin_v_conjugate_gradient_seq::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
