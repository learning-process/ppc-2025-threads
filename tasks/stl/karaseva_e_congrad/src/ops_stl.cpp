#include "stl/karaseva_e_congrad/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
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

template <typename Func>
void karaseva_a_test_task_stl::TestTaskSTL::Parallel(size_t start, size_t end, Func func) {
  // Manual parallelization with thread pool
  const size_t num_threads = std::thread::hardware_concurrency();
  const size_t chunk_size = (end - start + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; ++i) {
    const size_t chunk_start = start + i * chunk_size;
    const size_t chunk_end = std::min(end, chunk_start + chunk_size);
    if (chunk_start < chunk_end) {
      threads.emplace_back([=]() {
        for (size_t j = chunk_start; j < chunk_end; ++j) {
          func(j);
        }
      });
    }
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

bool karaseva_a_test_task_stl::TestTaskSTL::RunImpl() {
  // Initialize residual vector r = b - Ax (x is zero initially)
  std::vector<double> r = b_;
  std::vector<double> p = r;      // Initial search direction
  std::vector<double> ap(size_);  // Buffer for matrix-vector product

  // Calculate initial residual squared norm using parallel reduction
  double rs_old = 0.0;
  Parallel(0, size_, [&](size_t i) { rs_old += r[i] * r[i]; });

  const double tolerance = 1e-10;       // Convergence threshold
  const size_t max_iterations = size_;  // Max iterations (up to problem size)

  // Conjugate gradient main loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // Compute matrix-vector product: ap = A * p using parallel row processing
    Parallel(0, size_, [&](size_t i) {
      double sum = 0.0;
      for (size_t j = 0; j < size_; ++j) {
        sum += A_[i * size_ + j] * p[j];
      }
      ap[i] = sum;
    });

    // Calculate denominator for alpha: p^T * A * p using parallel reduction
    double p_ap = 0.0;
    Parallel(0, size_, [&](size_t i) { p_ap += p[i] * ap[i]; });

    if (std::fabs(p_ap) < 1e-15) {
      break;
    }
    const double alpha = rs_old / p_ap;

    // Parallel update of solution and residual vectors
    Parallel(0, size_, [&](size_t i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    });

    // Check convergence using residual norm
    double rs_new = 0.0;
    Parallel(0, size_, [&](size_t i) { rs_new += r[i] * r[i]; });

    if (rs_new < tolerance * tolerance) {
      break;
    }

    // Update search direction with beta parameter
    const double beta = rs_new / rs_old;
    Parallel(0, size_, [&](size_t i) { p[i] = r[i] + beta * p[i]; });

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