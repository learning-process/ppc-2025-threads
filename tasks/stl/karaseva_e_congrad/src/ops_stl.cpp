#include "stl/karaseva_e_congrad/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

bool karaseva_a_test_task_stl::TestTaskSTL::PreProcessingImpl() {
  // Set system size from input data
  size_ = task_data->inputs_count[1];
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Initialize matrix A and vectors b, x
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);
  x_ = std::vector<double>(size_, 0.0);  // Initial guess

  return true;
}

bool karaseva_a_test_task_stl::TestTaskSTL::ValidationImpl() {
  // Validate matrix and vector dimensions
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

bool karaseva_a_test_task_stl::TestTaskSTL::RunImpl() {
  std::vector<double> r(size_);
  std::vector<double> p(size_);
  std::vector<double> ap(size_);

  // Parallel initialization of residual and search direction vectors
  {
    const size_t num_threads =
        std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
      const size_t start = t * chunk_size;
      const size_t end = std::min(start + chunk_size, size_);
      // Explicit capture of 'this' for class member access
      threads.emplace_back([this, start, end, &r, &p]() {
        for (size_t i = start; i < end; ++i) {
          r[i] = this->b_[i];  // Access through captured 'this'
          p[i] = r[i];
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  double rs_old = 0.0;
  // Parallel reduction for initial residual norm
  {
    const size_t num_threads =
        std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
    std::vector<std::thread> threads(num_threads);
    std::vector<double> partial_sums(num_threads, 0.0);
    const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
      const size_t start = t * chunk_size;
      const size_t end = std::min(start + chunk_size, size_);
      threads[t] = std::thread([start, end, &r, &partial_sums, t]() {
        double sum = 0.0;
        for (size_t i = start; i < end; ++i) {
          sum += r[i] * r[i];
        }
        partial_sums[t] = sum;
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }

    for (const auto& sum : partial_sums) {
      rs_old += sum;
    }
  }

  const double tolerance = 1e-10;
  const size_t max_iterations = size_;

  // Main CG loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // Parallel matrix-vector multiplication
    {
      const size_t num_threads =
          std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
      std::vector<std::thread> threads;
      const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

      for (size_t t = 0; t < num_threads; ++t) {
        const size_t start = t * chunk_size;
        const size_t end = std::min(start + chunk_size, size_);
        // Capture 'this' for accessing matrix A_
        threads.emplace_back([this, start, end, &ap, &p]() {
          for (size_t i = start; i < end; ++i) {
            ap[i] = 0.0;
            for (size_t j = 0; j < this->size_; ++j) {
              ap[i] += this->A_[(i * this->size_) + j] * p[j];
            }
          }
        });
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }

    // Parallel dot product p^T * ap
    double p_ap = 0.0;
    {
      const size_t num_threads =
          std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
      std::vector<std::thread> threads(num_threads);
      std::vector<double> partial_sums(num_threads, 0.0);
      const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

      for (size_t t = 0; t < num_threads; ++t) {
        const size_t start = t * chunk_size;
        const size_t end = std::min(start + chunk_size, size_);
        threads[t] = std::thread([start, end, &p, &ap, &partial_sums, t]() {
          double sum = 0.0;
          for (size_t i = start; i < end; ++i) {
            sum += p[i] * ap[i];
          }
          partial_sums[t] = sum;
        });
      }

      for (auto& thread : threads) {
        thread.join();
      }

      for (const auto& sum : partial_sums) {
        p_ap += sum;
      }
    }

    if (std::fabs(p_ap) < 1e-15) break;
    const double alpha = rs_old / p_ap;

    // Parallel vector updates
    {
      const size_t num_threads =
          std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
      std::vector<std::thread> threads;
      const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

      for (size_t t = 0; t < num_threads; ++t) {
        const size_t start = t * chunk_size;
        const size_t end = std::min(start + chunk_size, size_);
        // Corrected capture list: use 'this' instead of '&x_'
        threads.emplace_back([this, start, end, alpha, &p, &r, &ap]() {
          for (size_t i = start; i < end; ++i) {
            this->x_[i] += alpha * p[i];  // Access x_ through 'this'
            r[i] -= alpha * ap[i];
          }
        });
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }

    // Convergence check
    double rs_new = 0.0;
    {
      const size_t num_threads =
          std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
      std::vector<std::thread> threads(num_threads);
      std::vector<double> partial_sums(num_threads, 0.0);
      const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

      for (size_t t = 0; t < num_threads; ++t) {
        const size_t start = t * chunk_size;
        const size_t end = std::min(start + chunk_size, size_);
        threads[t] = std::thread([start, end, &r, &partial_sums, t]() {
          double sum = 0.0;
          for (size_t i = start; i < end; ++i) {
            sum += r[i] * r[i];
          }
          partial_sums[t] = sum;
        });
      }

      for (auto& thread : threads) {
        thread.join();
      }

      for (const auto& sum : partial_sums) {
        rs_new += sum;
      }
    }

    if (rs_new < tolerance * tolerance) break;

    // Update search direction
    const double beta = rs_new / rs_old;
    {
      const size_t num_threads =
          std::max(static_cast<size_t>(1), static_cast<size_t>(std::thread::hardware_concurrency()));
      std::vector<std::thread> threads;
      const size_t chunk_size = (size_ + num_threads - 1) / num_threads;

      for (size_t t = 0; t < num_threads; ++t) {
        const size_t start = t * chunk_size;
        const size_t end = std::min(start + chunk_size, size_);
        threads.emplace_back([start, end, beta, &p, &r]() {
          for (size_t i = start; i < end; ++i) {
            p[i] = r[i] + beta * p[i];
          }
        });
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }

    rs_old = rs_new;
  }

  return true;
}

bool karaseva_a_test_task_stl::TestTaskSTL::PostProcessingImpl() {
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < x_.size(); ++i) {
    x_ptr[i] = x_[i];
  }
  return true;
}