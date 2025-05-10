#include "stl/karaseva_e_congrad/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <vector>

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
  // Initialize residual vector and search direction
  std::vector<double> r = b_;
  std::vector<double> p = r;
  std::vector<double> ap(size_);

  // Parallel computation of initial residual squared norm
  double rs_old = std::transform_reduce(std::execution::par,  // Parallel execution policy
                                        r.cbegin(), r.cend(), r.cbegin(), 0.0);

  const double tolerance = 1e-10;
  const size_t max_iterations = size_;

  // Generate indices for parallel matrix-vector product
  std::vector<size_t> indices(size_);
  std::iota(indices.begin(), indices.end(), 0);

  // Main conjugate gradient loop with parallel computations
  for (size_t k = 0; k < max_iterations; ++k) {
    // Parallel matrix-vector product using indices
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
      const auto row_start = A_.begin() + static_cast<std::ptrdiff_t>(i * size_);
      ap[i] = std::inner_product(row_start, row_start + static_cast<std::ptrdiff_t>(size_), p.cbegin(), 0.0);
    });

    // Parallel computation of p^T*A*p
    double p_ap = std::transform_reduce(std::execution::par,  // Parallel execution policy
                                        p.cbegin(), p.cend(), ap.cbegin(), 0.0);

    if (std::fabs(p_ap) < 1e-15) {
      break;
    }
    const double alpha = rs_old / p_ap;

    // Parallel vector updates using transform
    std::transform(std::execution::par,  // Parallel execution
                   x_.cbegin(), x_.cend(), p.cbegin(), x_.begin(),
                   [alpha](double x, double p_val) { return x + (alpha * p_val); });

    std::transform(std::execution::par,  // Parallel execution
                   r.cbegin(), r.cend(), ap.cbegin(), r.begin(),
                   [alpha](double r_val, double ap_val) { return r_val - (alpha * ap_val); });

    // Parallel residual norm calculation
    double rs_new = std::transform_reduce(std::execution::par,  // Parallel execution policy
                                          r.cbegin(), r.cend(), r.cbegin(), 0.0);

    if (rs_new < tolerance * tolerance) {
      break;
    }

    const double beta = rs_new / rs_old;

    // Parallel direction update
    std::transform(std::execution::par,  // Parallel execution
                   r.cbegin(), r.cend(), p.cbegin(), p.begin(),
                   [beta](double r_val, double p_val) { return r_val + (beta * p_val); });

    rs_old = rs_new;
  }

  return true;
}

bool karaseva_a_test_task_stl::TestTaskSTL::PostProcessingImpl() {
  // Copy solution vector to output buffer
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(x_, x_ptr);
  return true;
}