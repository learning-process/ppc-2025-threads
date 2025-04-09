#include "tbb/karaseva_e_congrad_tbb/include/ops_tbb.hpp"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <vector>

bool karaseva_e_congrad_tbb::TestTaskTBB::PreProcessingImpl() {
  // Initialize problem size from input data
  size_ = task_data->inputs_count[1];

  // Map raw input pointers to matrices/vectors
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Create contiguous storage for matrix A and vector b
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);

  // Initial guess x0 = 0
  x_ = std::vector<double>(size_, 0.0);

  return true;
}

bool karaseva_e_congrad_tbb::TestTaskTBB::ValidationImpl() {
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

bool karaseva_e_congrad_tbb::TestTaskTBB::RunImpl() {
  // Residual vector (r = b - Ax)
  std::vector<double> r(size_);

  // Search direction vector
  std::vector<double> p(size_);

  // Matrix-vector product (Ap)
  std::vector<double> ap(size_);

  // Parallel initialization of r and p vectors
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      r[i] = b_[i];
      p[i] = r[i];
    }
  });

  // Compute initial residual squared norm (rs_old = r^T * r)
  double rs_old = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, size_), 0.0,
      [&](const tbb::blocked_range<size_t>& range, double local_sum) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          local_sum += r[i] * r[i];
        }
        return local_sum;
      },
      std::plus<>());

  // Convergence tolerance and iteration limit
  const double tolerance = 1e-10;
  const size_t max_iterations = size_;

  // Main conjugate gradient loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // Parallel matrix-vector multiplication: ap = A * p
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        double temp = 0.0;
        for (size_t j = 0; j < size_; ++j) {
          temp += A_[(i * size_) + j] * p[j];
        }
        ap[i] = temp;
      }
    });

    // Compute p^T * A * p (denominator for alpha)
    double p_ap = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, size_), 0.0,
        [&](const tbb::blocked_range<size_t>& range, double local_sum) {
          for (size_t i = range.begin(); i != range.end(); ++i) {
            local_sum += p[i] * ap[i];
          }
          return local_sum;
        },
        std::plus<>());

    // Avoid division by zero
    if (std::fabs(p_ap) < 1e-15) break;

    // Compute step size alpha
    const double alpha = rs_old / p_ap;

    // Parallel update of solution and residual vectors
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        x_[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
    });

    // Compute new residual squared norm
    double rs_new = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, size_), 0.0,
        [&](const tbb::blocked_range<size_t>& range, double local_sum) {
          for (size_t i = range.begin(); i != range.end(); ++i) {
            local_sum += r[i] * r[i];
          }
          return local_sum;
        },
        std::plus<>());

    // Check convergence
    if (rs_new < tolerance * tolerance) break;

    // Compute beta for direction update
    const double beta = rs_new / rs_old;

    // Parallel update of search direction
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size_), [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        p[i] = r[i] + beta * p[i];
      }
    });

    rs_old = rs_new;
  }

  return true;
}

bool karaseva_e_congrad_tbb::TestTaskTBB::PostProcessingImpl() {
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, x_.size()), [&](const tbb::blocked_range<size_t>& range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      x_ptr[i] = x_[i];
    }
  });
  return true;
}