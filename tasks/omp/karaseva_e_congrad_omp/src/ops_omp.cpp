#include "omp/karaseva_e_congrad_omp/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool karaseva_e_congrad_omp::TestTaskOpenMP::PreProcessingImpl() {
  // Initialize problem dimensions and input data
  size_ = task_data->inputs_count[1];
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Copy input matrix A and vector b to internal storage
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);
  x_ = std::vector<double>(size_, 0.0);

  return true;
}

bool karaseva_e_congrad_omp::TestTaskOpenMP::ValidationImpl() {
  // Validate that input matrix is square (size^2 elements)
  // and output vector has correct size
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

bool karaseva_e_congrad_omp::TestTaskOpenMP::RunImpl() {
  // Conjugate Gradient algorithm implementation
  std::vector<double> r(size_);
  std::vector<double> p(size_);
  std::vector<double> ap(size_);

  // Parallel initialization of residual (r) and search direction (p) vectors
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(size_); ++i) {
    r[i] = b_[i];
    p[i] = r[i];
  }

  // Compute initial residual squared norm
  double rs_old = 0.0;
#pragma omp parallel for reduction(+ : rs_old)
  for (int i = 0; i < static_cast<int>(size_); ++i) {
    rs_old += r[i] * r[i];
  }

  const double tolerance = 1e-10;
  const size_t max_iterations = size_;

  // Main conjugate gradient iteration loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // Parallel matrix-vector multiplication: ap = A * p
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      ap[i] = 0.0;
      for (size_t j = 0; j < size_; ++j) {
        ap[i] += A_[(i * size_) + j] * p[j];
      }
    }

    // Compute inner product for alpha calculation
    double p_ap = 0.0;
#pragma omp parallel for reduction(+ : p_ap)
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      p_ap += p[i] * ap[i];
    }

    // Early exit if denominator becomes too small
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }
    const double alpha = rs_old / p_ap;

    // Parallel update of solution (x) and residual (r) vectors
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }

    // Compute new residual squared norm
    double rs_new = 0.0;
#pragma omp parallel for reduction(+ : rs_new)
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      rs_new += r[i] * r[i];
    }

    // Check convergence condition
    if (rs_new < tolerance * tolerance) {
      break;
    }

    // Compute beta for direction vector update
    const double beta = rs_new / rs_old;

    // Parallel update of search direction vector (p)
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(size_); ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }

  return true;
}

bool karaseva_e_congrad_omp::TestTaskOpenMP::PostProcessingImpl() {
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(x_.size()); ++i) {
    x_ptr[i] = x_[i];
  }
  return true;
}