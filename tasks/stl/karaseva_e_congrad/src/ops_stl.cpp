#include "stl/karaseva_e_congrad/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool karaseva_a_test_task_stl::TestTaskSTL::PreProcessingImpl() {
  // Initialize problem size from input vector length
  size_ = task_data->inputs_count[1];

  // Map input pointers to matrix A and vector b
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Construct matrix A (size_ x size_) and vector b from input data
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);
  x_ = std::vector<double>(size_, 0.0);

  return true;
}

bool karaseva_a_test_task_stl::TestTaskSTL::ValidationImpl() {
  // Validate matrix dimensions: A should be square matrix (n x n)
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  // Validate output vector size matches problem dimension
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

bool karaseva_a_test_task_stl::TestTaskSTL::RunImpl() {
  // Initialize residual vector r = b - Ax (x is zero initially)
  std::vector<double> r = b_;
  std::vector<double> p = r;      // Initial search direction
  std::vector<double> ap(size_);  // Buffer for matrix-vector product

  // Calculate initial residual squared norm
  double rs_old = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

  const double tolerance = 1e-10;       // Convergence threshold
  const size_t max_iterations = size_;  // Max iterations (up to problem size)

  // Conjugate gradient main loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // Compute matrix-vector product: ap = A * p
    for (size_t i = 0; i < size_; ++i) {
      auto row_start = A_.begin() + i * size_;
      ap[i] = std::inner_product(row_start, row_start + size_, p.begin(), 0.0);
    }

    // Calculate denominator for alpha: p^T * A * p
    double p_ap = std::inner_product(p.begin(), p.end(), ap.begin(), 0.0);
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }
    const double alpha = rs_old / p_ap;

    // Update solution and residual vectors
    std::transform(x_.begin(), x_.end(), p.begin(), x_.begin(),
                   [alpha](double x_i, double p_i) { return x_i + alpha * p_i; });
    std::transform(r.begin(), r.end(), ap.begin(), r.begin(),
                   [alpha](double r_i, double ap_i) { return r_i - alpha * ap_i; });

    // Check convergence using residual norm
    double rs_new = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
    if (rs_new < tolerance * tolerance) {
      break;
    }

    // Update search direction with beta parameter
    const double beta = rs_new / rs_old;
    std::transform(r.begin(), r.end(), p.begin(), p.begin(),
                   [beta](double r_i, double p_i) { return r_i + beta * p_i; });

    rs_old = rs_new;
  }

  return true;
}

bool karaseva_a_test_task_stl::TestTaskSTL::PostProcessingImpl() {
  // Copy solution vector to output buffer
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(x_.begin(), x_.end(), x_ptr);
  return true;
}