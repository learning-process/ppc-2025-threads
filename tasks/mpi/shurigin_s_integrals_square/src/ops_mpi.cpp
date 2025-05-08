#include "mpi/shurigin_s_integrals_square/include/ops_mpi.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shurigin_s_integrals_square_mpi {

Integral::Integral(std::shared_ptr<ppc::core::TaskData> task_data_param)
    : Task(task_data_param),
      task_data_(task_data_param),
      down_limits_(1, 0.0),
      up_limits_(1, 0.0),
      counts_(1, 0),
      result_(0.0),
      func_(nullptr),
      dimensions_(1),
      mpi_rank_(0),
      mpi_world_size_(1) {}

void Integral::SetFunction(const std::function<double(double)>& func) {
  func_ = [func](const std::vector<double>& point) {
    if (point.empty()) {
      throw std::runtime_error("Internal error: Point vector is empty in 1D wrapper.");
    }
    return func(point[0]);
  };
  dimensions_ = 1;
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions) {
  if (dimensions <= 0) {
    throw std::invalid_argument("Dimensions must be positive.");
  }
  func_ = func;
  dimensions_ = dimensions;
}

bool Integral::PreProcessingImpl() {
  try {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size_);

    if (mpi_rank_ == 0) {
      if (!this->task_data_ || this->task_data_->inputs.empty() || this->task_data_->inputs[0] == nullptr) {
        throw std::invalid_argument("Invalid input data on root.");
      }
    }

    MPI_Bcast(&dimensions_, 1, MPI_INT, 0, MPI_COMM_WORLD);

    down_limits_.resize(dimensions_);
    up_limits_.resize(dimensions_);
    counts_.resize(dimensions_);

    if (mpi_rank_ == 0) {
      double* inputs_ptr = reinterpret_cast<double*>(this->task_data_->inputs[0]);
      if (!inputs_ptr) {
        throw std::runtime_error("Root: input data pointer (inputs[0]) is null.");
      }
      size_t expected_input_size_bytes = static_cast<size_t>(3 * dimensions_) * sizeof(double);
      if (this->task_data_->inputs_count[0] != expected_input_size_bytes) {
        throw std::invalid_argument("Root: Input data size mismatch. Expected " +
                                    std::to_string(expected_input_size_bytes) + " bytes, got " +
                                    std::to_string(this->task_data_->inputs_count[0]) + " bytes.");
      }

      for (int i = 0; i < dimensions_; ++i) {
        down_limits_[i] = inputs_ptr[i];
        up_limits_[i] = inputs_ptr[i + dimensions_];
        counts_[i] = static_cast<int>(inputs_ptr[i + (2 * dimensions_)]);

        if (counts_[i] <= 0) {
          throw std::invalid_argument("Root: Number of intervals must be positive for all dimensions.");
        }
        if (up_limits_[i] <= down_limits_[i]) {
          throw std::invalid_argument("Root: Upper limit must be greater than lower limit for all dimensions.");
        }
      }
    }

    MPI_Bcast(down_limits_.data(), dimensions_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(up_limits_.data(), dimensions_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(counts_.data(), dimensions_, MPI_INT, 0, MPI_COMM_WORLD);

    result_ = 0.0;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in PreProcessingImpl: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    int validation_success_int = 1;

    if (mpi_rank_ == 0) {
      if (!this->task_data_) {
        try {
          throw std::invalid_argument("task_data_ is null on root.");
        } catch (const std::exception& e_local) {
          validation_success_int = 0;
          std::cerr << "Rank 0 ValidationImpl Error: " << e_local.what() << std::endl;
        }
      }
      if (validation_success_int &&
          (this->task_data_->inputs_count.empty() || this->task_data_->outputs_count.empty())) {
        try {
          throw std::invalid_argument("Input or output counts are empty on root.");
        } catch (const std::exception& e_local) {
          validation_success_int = 0;
          std::cerr << "Rank 0 ValidationImpl Error: " << e_local.what() << std::endl;
        }
      }

      if (validation_success_int) {
        size_t expected_input_size = 3 * dimensions_ * sizeof(double);
        if (this->task_data_->inputs_count[0] != expected_input_size) {
          try {
            throw std::invalid_argument("Root: Input data size validation failed.");
          } catch (const std::exception& e_local) {
            validation_success_int = 0;
            std::cerr << "Rank 0 ValidationImpl Error: " << e_local.what() << " Expected " << expected_input_size
                      << ", got " << this->task_data_->inputs_count[0] << std::endl;
          }
        }
      }

      if (validation_success_int) {
        if (this->task_data_->outputs_count[0] != sizeof(double)) {
          try {
            throw std::invalid_argument("Root: Output data size validation failed.");
          } catch (const std::exception& e_local) {
            validation_success_int = 0;
            std::cerr << "Rank 0 ValidationImpl Error: " << e_local.what() << " Expected " << sizeof(double) << ", got "
                      << this->task_data_->outputs_count[0] << std::endl;
          }
        }
      }
    }

    MPI_Bcast(&validation_success_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool final_validation_success = (validation_success_int == 1);

    if (!final_validation_success && mpi_rank_ != 0) {
      std::cerr << "Rank " << mpi_rank_ << " ValidationImpl: Received validation failure from root." << std::endl;
    }
    if (final_validation_success && dimensions_ <= 0) {
      throw std::logic_error("Rank " + std::to_string(mpi_rank_) + ": Dimensions not set or invalid (" +
                             std::to_string(dimensions_) + ") despite root validation attempt.");
    }

    return final_validation_success;

  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in ValidationImpl (local exception): " << e.what() << '\n';
    int local_failure = 0;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("Function is not set on rank " + std::to_string(mpi_rank_));
    }

    double local_result = 0.0;

    if (dimensions_ <= 0 || counts_.empty()) {
      if (dimensions_ <= 0) {
        throw std::logic_error("RunImpl: dimensions_ is invalid (" + std::to_string(dimensions_) + ") on rank " +
                               std::to_string(mpi_rank_));
      }
      if (counts_.empty()) {
        throw std::logic_error("RunImpl: counts_ vector is empty on rank " + std::to_string(mpi_rank_));
      }
    }
    int N0 = counts_[0];

    if (N0 > 0) {
      double a0_full = down_limits_[0];
      double b0_full = up_limits_[0];
      double h0_full_step = (N0 > 0) ? (b0_full - a0_full) / N0 : 0;

      int chunk_size = N0 / mpi_world_size_;
      int remainder = N0 % mpi_world_size_;
      int local_n0_start_idx = mpi_rank_ * chunk_size + std::min(mpi_rank_, remainder);
      int local_n0_end_idx = local_n0_start_idx + chunk_size + (mpi_rank_ < remainder ? 1 : 0);

      int local_n0 = local_n0_end_idx - local_n0_start_idx;
      double local_a0 = a0_full + local_n0_start_idx * h0_full_step;
      double local_b0 = local_a0 + local_n0 * h0_full_step;

      if (mpi_rank_ == mpi_world_size_ - 1) {
        local_b0 = b0_full;
      }
      if (local_n0 > 0 && local_b0 > b0_full) {
        if (std::abs(local_b0 - b0_full) > 1e-9 * std::abs(b0_full)) {
          std::cerr << "Warning Rank " << mpi_rank_ << ": local_b0 (" << local_b0 << ") exceeded b0_full (" << b0_full
                    << "). Clamping." << std::endl;
        }
        local_b0 = b0_full;
      }
      if (local_n0 > 0 && local_a0 < a0_full) {
        if (std::abs(local_a0 - a0_full) > 1e-9 * std::abs(a0_full)) {
          std::cerr << "Warning Rank " << mpi_rank_ << ": local_a0 (" << local_a0 << ") was less than a0_full ("
                    << a0_full << "). Clamping." << std::endl;
        }
        local_a0 = a0_full;
      }
      if (local_a0 >= local_b0) {
        local_n0 = 0;
      }

      if (local_n0 > 0) {
        if (dimensions_ == 1) {
          local_result = ComputeOneDimensionalOMP(func_, local_a0, local_b0, local_n0);
        } else {
          local_result = ComputeOuterParallelInnerSequential(func_, local_a0, local_b0, local_n0, down_limits_,
                                                             up_limits_, counts_, dimensions_);
        }
      }
    }

    MPI_Reduce(&local_result, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in RunImpl: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

double Integral::ComputeOneDimensionalOMP(const std::function<double(const std::vector<double>&)>& f, double a_local,
                                          double b_local, int n_local) {
  if (n_local <= 0 || a_local >= b_local) {
    return 0.0;
  }

  const double step = (b_local - a_local) / n_local;
  const double half_step = 0.5 * step;
  double sum = 0.0;

#pragma omp parallel
  {
    std::vector<double> point(1);
#pragma omp for reduction(+ : sum)
    for (int i = 0; i < n_local; ++i) {
      point[0] = a_local + (static_cast<double>(i) + 0.5) * step;
      sum += f(point);
    }
  }
  return sum * step;
}

double Integral::ComputeOuterParallelInnerSequential(const std::function<double(const std::vector<double>&)>& f,
                                                     double a0_local, double b0_local, int n0_local,
                                                     const std::vector<double>& full_a,
                                                     const std::vector<double>& full_b, const std::vector<int>& full_n,
                                                     int total_dims) {
  if (n0_local <= 0 || a0_local >= b0_local) {
    return 0.0;
  }

  const double h0_local = (b0_local - a0_local) / n0_local;
  double outer_sum = 0.0;

#pragma omp parallel
  {
    std::vector<double> point(total_dims);
#pragma omp for reduction(+ : outer_sum)
    for (int i = 0; i < n0_local; ++i) {
      point[0] = a0_local + (static_cast<double>(i) + 0.5) * h0_local;
      if (total_dims == 1) {
        outer_sum += f(point);
      } else {
        outer_sum += ComputeSequentialRecursive(f, full_a, full_b, full_n, total_dims, point, 1);
      }
    }
  }
  return outer_sum * h0_local;
}

double Integral::ComputeSequentialRecursive(const std::function<double(const std::vector<double>&)>& f,
                                            const std::vector<double>& a, const std::vector<double>& b,
                                            const std::vector<int>& n, int total_dims, std::vector<double>& point,
                                            int current_dim_idx) {
  if (current_dim_idx == total_dims) {
    return f(point);
  }

  if (current_dim_idx < 0 || static_cast<size_t>(current_dim_idx) >= n.size() ||
      static_cast<size_t>(current_dim_idx) >= a.size() || static_cast<size_t>(current_dim_idx) >= b.size()) {
    throw std::out_of_range("Dimension index " + std::to_string(current_dim_idx) +
                            " out of bounds in ComputeSequentialRecursive.");
  }

  const int current_n = n[current_dim_idx];
  const double current_a = a[current_dim_idx];
  const double current_b = b[current_dim_idx];

  if (current_n <= 0 || current_a >= current_b) {
    if (current_n <= 0) {
      throw std::runtime_error("Internal error: Non-positive interval count (N=" + std::to_string(current_n) +
                               ") for dim " + std::to_string(current_dim_idx));
    }
    return 0.0;
  }

  const double step = (current_b - current_a) / current_n;
  double integral_sum_for_this_dim = 0.0;
  for (int i = 0; i < current_n; ++i) {
    point[current_dim_idx] = current_a + (static_cast<double>(i) + 0.5) * step;
    integral_sum_for_this_dim += ComputeSequentialRecursive(f, a, b, n, total_dims, point, current_dim_idx + 1);
  }

  return integral_sum_for_this_dim * step;
}

bool Integral::PostProcessingImpl() {
  try {
    if (mpi_rank_ == 0) {
      if (!this->task_data_ || this->task_data_->outputs.empty() || this->task_data_->outputs[0] == nullptr) {
        throw std::invalid_argument("Invalid output data on root.");
      }
      double* outputs_ptr = reinterpret_cast<double*>(this->task_data_->outputs[0]);
      if (!outputs_ptr) {
        throw std::runtime_error("Root: output data pointer (outputs[0]) is null.");
      }
      outputs_ptr[0] = result_;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in PostProcessingImpl: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

}  // namespace shurigin_s_integrals_square_mpi