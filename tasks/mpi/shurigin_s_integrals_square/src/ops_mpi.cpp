#include "mpi/shurigin_s_integrals_square/include/ops_mpi.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace shurigin_s_integrals_square_mpi {

Integral::Integral(const std::shared_ptr<ppc::core::TaskData>& task_data_param)
    : Task(task_data_param),
      down_limits_(1, 0.0),
      up_limits_(1, 0.0),
      counts_(1, 0),
      result_(0.0),
      func_(nullptr),
      dimensions_(1),
      task_data_(task_data_param),
      mpi_rank_(0),
      mpi_world_size_(1) {}

void Integral::SetFunction(const std::function<double(double)>& func) {
  if (!func) {
    throw std::invalid_argument("Function provided is null.");
  }
  func_ = [func](const std::vector<double>& point) {
    if (point.empty()) {
      throw std::runtime_error("Internal error: Point vector is empty in 1D wrapper.");
    }
    return func(point[0]);
  };
  dimensions_ = 1;
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions) {
  if (!func) {
    throw std::invalid_argument("Function provided is null.");
  }
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
        throw std::invalid_argument("Invalid input task data or input buffer on root.");
      }
      size_t expected_input_size_bytes = static_cast<size_t>(3 * dimensions_) * sizeof(double);
      if (this->task_data_->inputs_count.empty() || this->task_data_->inputs_count[0] != expected_input_size_bytes) {
        throw std::invalid_argument("Input data size mismatch on root.");
      }
    }

    MPI_Bcast(&dimensions_, 1, MPI_INT, 0, MPI_COMM_WORLD);

    down_limits_.resize(dimensions_);
    up_limits_.resize(dimensions_);
    counts_.resize(dimensions_);

    std::vector<double> all_data_buffer(static_cast<size_t>(3 * dimensions_));

    if (mpi_rank_ == 0) {
      auto* inputs_ptr = reinterpret_cast<double*>(this->task_data_->inputs[0]);
      for (int i = 0; i < dimensions_; ++i) {
        down_limits_[i] = inputs_ptr[i];
        up_limits_[i] = inputs_ptr[i + dimensions_];
        counts_[i] = static_cast<int>(inputs_ptr[i + (2 * dimensions_)]);

        if (counts_[i] <= 0) {
          throw std::invalid_argument("Number of intervals must be positive.");
        }
        if (up_limits_[i] <= down_limits_[i]) {
          throw std::invalid_argument("Upper limit must be greater than lower limit.");
        }
        all_data_buffer[static_cast<size_t>(i)] = down_limits_[i];
        all_data_buffer[static_cast<size_t>(i + dimensions_)] = up_limits_[i];
        all_data_buffer[static_cast<size_t>(i + (2 * dimensions_))] = static_cast<double>(counts_[i]);
      }
    }
    MPI_Bcast(all_data_buffer.data(), static_cast<int>(all_data_buffer.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank_ != 0) {
      for (int i = 0; i < dimensions_; ++i) {
        down_limits_[i] = all_data_buffer[static_cast<size_t>(i)];
        up_limits_[i] = all_data_buffer[static_cast<size_t>(i + dimensions_)];
        counts_[i] = static_cast<int>(all_data_buffer[static_cast<size_t>(i + (2 * dimensions_))]);
      }
    }

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
    int validation_status = 1;

    if (mpi_rank_ == 0) {
      bool root_data_valid = true;
      if (!this->task_data_) {
        root_data_valid = false;
      } else if (this->task_data_->inputs_count.empty() || this->task_data_->outputs_count.empty()) {
        root_data_valid = false;
      } else {
        size_t expected_input_size = static_cast<size_t>(3 * dimensions_) * sizeof(double);
        if (this->task_data_->inputs_count[0] != expected_input_size) {
          root_data_valid = false;
        }
        if (this->task_data_->outputs_count[0] != sizeof(double)) {
          root_data_valid = false;
        }
      }

      if (!root_data_valid) {
        validation_status = 0;
        std::cerr << "Rank 0 ValidationImpl Error: Basic task data validation failed." << '\n';
      }
    }

    MPI_Bcast(&validation_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool is_valid = (validation_status == 1);

    if (is_valid && dimensions_ <= 0) {
      throw std::runtime_error("Validation failed: Invalid dimensions after Bcast.");
    }
    if (is_valid && !func_) {
      throw std::runtime_error("Validation failed: Integration function is not set.");
    }

    return is_valid;

  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in ValidationImpl: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("RunImpl: Function is not set.");
    }

    double local_result = 0.0;

    if (dimensions_ <= 0 || counts_.empty()) {
      throw std::runtime_error("RunImpl: Invalid dimensions or counts vector.");
    }
    int n0 = counts_[0];

    if (n0 > 0) {
      double a0_full = down_limits_[0];
      double b0_full = up_limits_[0];
      double h0_full_step = (b0_full - a0_full) / n0;

      int chunk_size = n0 / mpi_world_size_;
      int remainder = n0 % mpi_world_size_;
      int local_n0_start_idx = (mpi_rank_ * chunk_size) + std::min(mpi_rank_, remainder);
      int local_n0_end_idx = local_n0_start_idx + chunk_size + (mpi_rank_ < remainder ? 1 : 0);

      int local_n0 = local_n0_end_idx - local_n0_start_idx;
      double local_a0 = a0_full + (static_cast<double>(local_n0_start_idx) * h0_full_step);
      double local_b0 = local_a0 + (static_cast<double>(local_n0) * h0_full_step);

      if (mpi_rank_ == mpi_world_size_ - 1) {
        local_b0 = b0_full;
      }
      local_a0 = std::max(a0_full, local_a0);
      local_b0 = std::min(b0_full, local_b0);

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
  double total_sum = 0.0;

#pragma omp parallel
  {
    std::vector<double> point(1);
    double private_sum = 0.0;
#pragma omp for nowait
    for (int i = 0; i < n_local; ++i) {
      point[0] = a_local + (static_cast<double>(i) + 0.5) * step;
      private_sum += f(point);
    }
#pragma omp critical
    total_sum += private_sum;
  }
  return total_sum * step;
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
  double outer_total_sum = 0.0;

#pragma omp parallel
  {
    std::vector<double> point(static_cast<size_t>(total_dims));
    double private_sum = 0.0;
#pragma omp for nowait
    for (int i = 0; i < n0_local; ++i) {
      point[0] = a0_local + (static_cast<double>(i) + 0.5) * h0_local;
      private_sum += ComputeSequentialRecursive(f, full_a, full_b, full_n, total_dims, point, 1);
    }
#pragma omp critical
    outer_total_sum += private_sum;
  }
  return outer_total_sum * h0_local;
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
    throw std::out_of_range("Dimension index out of bounds in recursive call.");
  }

  const int current_n = n[static_cast<size_t>(current_dim_idx)];
  const double current_a = a[static_cast<size_t>(current_dim_idx)];
  const double current_b = b[static_cast<size_t>(current_dim_idx)];

  if (current_n <= 0) {
    throw std::runtime_error("Non-positive interval count in recursive call.");
  }
  if (current_a >= current_b) {
    return 0.0;
  }

  const double step = (current_b - current_a) / current_n;
  double integral_sum_for_this_dim = 0.0;

  for (int i = 0; i < current_n; ++i) {
    point[static_cast<size_t>(current_dim_idx)] = current_a + (static_cast<double>(i) + 0.5) * step;
    integral_sum_for_this_dim += ComputeSequentialRecursive(f, a, b, n, total_dims, point, current_dim_idx + 1);
  }

  return integral_sum_for_this_dim * step;
}

bool Integral::PostProcessingImpl() {
  try {
    if (mpi_rank_ == 0) {
      if (!this->task_data_ || this->task_data_->outputs.empty() || this->task_data_->outputs[0] == nullptr) {
        throw std::invalid_argument("Invalid output task data or output buffer on root.");
      }
      if (this->task_data_->outputs_count.empty() || this->task_data_->outputs_count[0] != sizeof(double)) {
        throw std::invalid_argument("Output data size mismatch on root.");
      }

      auto* outputs_ptr = reinterpret_cast<double*>(this->task_data_->outputs[0]);
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