#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"
#endif

#include "mpi/shurigin_s_integrals_square/include/ops_mpi.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace shurigin_s_integrals_square_mpi {

Integral::Integral(const std::shared_ptr<ppc::core::TaskData>& task_data_param)
    : Task(task_data_param),
      down_limits_(),
      up_limits_(),
      counts_(),
      result_(0.0),
      func_(nullptr),
      dimensions_(0),
      mpi_rank_(0),
      mpi_world_size_(1) {}

void Integral::SetFunction(const std::function<double(double)>& func) {
  if (!func) {
    throw std::invalid_argument("Function provided is null (1D).");
  }
  func_ = [func](const std::vector<double>& point) {
    if (point.empty() || point.size() < 1) {
      throw std::runtime_error("Internal error: Point vector is empty or too small in 1D wrapper.");
    }
    return func(point[0]);
  };
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dims_param) {
  if (!func) {
    throw std::invalid_argument("Function provided is null (ND).");
  }
  if (dims_param <= 0) {
    throw std::invalid_argument("Dimensions must be positive.");
  }
  func_ = func;
  dimensions_ = dims_param;
}

bool Integral::PreProcessingImpl() {
  try {
    if (!this->taskData) {
      throw std::invalid_argument("TaskData is not set (null).");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size_);

    int temp_dimensions = 0;

    if (mpi_rank_ == 0) {
      if (this->taskData->inputs.empty() || this->taskData->inputs[0] == nullptr ||
          this->taskData->inputs_count.empty()) {
        throw std::invalid_argument("Invalid input task data or input buffer on root process.");
      }

      if (dimensions_ <= 0 && this->taskData->inputs_count[0] >= 3 * sizeof(double)) {
        size_t num_elements = this->taskData->inputs_count[0] / sizeof(double);
        if (num_elements % 3 != 0 || num_elements == 0) {
          throw std::invalid_argument(
              "Cannot determine dimensions from input data size on root. Size is not a multiple of 3 or is zero.");
        }
        temp_dimensions = static_cast<int>(num_elements / 3);
        if (temp_dimensions <= 0) {
          throw std::invalid_argument("Deduced dimensions are not positive from input data size on root.");
        }
        dimensions_ = temp_dimensions;
        std::cout << "Rank 0: Deduced dimensions = " << dimensions_ << " from input data." << std::endl;
      } else if (dimensions_ <= 0) {
        throw std::invalid_argument(
            "Dimensions not set prior to PreProcessing and cannot be deduced (input size too small or dimensions_ was "
            "0).");
      }
      temp_dimensions = dimensions_;

      size_t expected_num_elements = static_cast<size_t>(3 * temp_dimensions);
      size_t actual_num_elements = this->taskData->inputs_count[0] / sizeof(double);

      if (actual_num_elements != expected_num_elements) {
        throw std::invalid_argument(
            "Input data size mismatch on root. Expected " + std::to_string(expected_num_elements) + " doubles (for " +
            std::to_string(temp_dimensions) + " dimensions), got " + std::to_string(actual_num_elements) + " doubles.");
      }

      down_limits_.resize(temp_dimensions);
      up_limits_.resize(temp_dimensions);
      counts_.resize(temp_dimensions);

      auto* inputs_ptr = reinterpret_cast<double*>(this->taskData->inputs[0]);
      for (int i = 0; i < temp_dimensions; ++i) {
        down_limits_[i] = inputs_ptr[i];
        up_limits_[i] = inputs_ptr[i + temp_dimensions];
        counts_[i] = static_cast<int>(inputs_ptr[i + (2 * temp_dimensions)]);

        if (counts_[i] <= 0) {
          throw std::invalid_argument("Number of intervals (counts) must be positive for dimension " +
                                      std::to_string(i));
        }
        if (up_limits_[i] <= down_limits_[i]) {
          throw std::invalid_argument("Upper limit must be greater than lower limit for dimension " +
                                      std::to_string(i));
        }
      }
    }

    MPI_Bcast(&dimensions_, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank_ != 0) {
      if (dimensions_ <= 0) {
        throw std::runtime_error("Rank " + std::to_string(mpi_rank_) + ": Received invalid dimensions_ (" +
                                 std::to_string(dimensions_) + ") from Bcast.");
      }
      down_limits_.resize(dimensions_);
      up_limits_.resize(dimensions_);
      counts_.resize(dimensions_);
    }

    MPI_Bcast(down_limits_.data(), dimensions_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(up_limits_.data(), dimensions_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(counts_.data(), dimensions_, MPI_INT, 0, MPI_COMM_WORLD);

    result_ = 0.0;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in PreProcessingImpl: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    if (!this->taskData) {
      if (mpi_rank_ == 0) std::cerr << "Rank 0 ValidationImpl Error: TaskData is null." << std::endl;
      return false;
    }

    if (dimensions_ <= 0) {
      std::cerr << "Rank " << mpi_rank_ << " ValidationImpl Error: dimensions_ is not positive (" << dimensions_ << ")."
                << std::endl;
      return false;
    }

    int global_validation_status = 1;

    if (mpi_rank_ == 0) {
      bool root_data_valid = true;
      if (this->taskData->inputs_count.empty() || this->taskData->outputs_count.empty()) {
        root_data_valid = false;
        std::cerr << "Rank 0 ValidationImpl Error: inputs_count or outputs_count is empty." << std::endl;
      } else {
        size_t expected_num_elements_input = static_cast<size_t>(3 * dimensions_);
        size_t actual_num_elements_input = this->taskData->inputs_count[0] / sizeof(double);

        if (actual_num_elements_input != expected_num_elements_input) {
          root_data_valid = false;
          std::cerr << "Rank 0 ValidationImpl Error: Input data size mismatch. Expected " << expected_num_elements_input
                    << " doubles, got " << actual_num_elements_input << " doubles." << std::endl;
        }
        if (this->taskData->outputs_count[0] != sizeof(double)) {
          root_data_valid = false;
          std::cerr << "Rank 0 ValidationImpl Error: Output data size mismatch. Expected " << sizeof(double)
                    << " bytes, got " << this->taskData->outputs_count[0] << " bytes." << std::endl;
        }
      }
      if (!root_data_valid) {
        global_validation_status = 0;
      }
    }

    MPI_Bcast(&global_validation_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (global_validation_status == 0) {
      if (mpi_rank_ != 0)
        std::cerr << "Rank " << mpi_rank_ << " ValidationImpl Error: Root process reported validation failure."
                  << std::endl;
      return false;
    }

    if (!func_) {
      std::cerr << "Rank " << mpi_rank_ << " ValidationImpl Error: Integration function is not set." << std::endl;
      global_validation_status = 0;
    }
    if (static_cast<int>(down_limits_.size()) != dimensions_ || static_cast<int>(up_limits_.size()) != dimensions_ ||
        static_cast<int>(counts_.size()) != dimensions_) {
      std::cerr << "Rank " << mpi_rank_
                << " ValidationImpl Error: Mismatch in vector sizes after PreProcessing for dimensions_ = "
                << dimensions_ << ". down_limits: " << down_limits_.size() << ", up_limits: " << up_limits_.size()
                << ", counts: " << counts_.size() << std::endl;
      global_validation_status = 0;
    }

    int final_status;
    MPI_Allreduce(&global_validation_status, &final_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    return (final_status == 1);

  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in ValidationImpl: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("RunImpl Error: Function is not set.");
    }
    if (dimensions_ <= 0 || counts_.empty() || down_limits_.empty() || up_limits_.empty()) {
      throw std::runtime_error("RunImpl Error: Invalid dimensions or improperly initialized limits/counts vectors.");
    }
    if (static_cast<int>(counts_.size()) != dimensions_ || static_cast<int>(down_limits_.size()) != dimensions_ ||
        static_cast<int>(up_limits_.size()) != dimensions_) {
      throw std::runtime_error("RunImpl Error: Mismatch between dimensions_ and sizes of limits/counts vectors.");
    }

    double local_integral_sum = 0.0;
    int n0_total = counts_[0];

    if (n0_total > 0) {
      double a0_global = down_limits_[0];
      double b0_global = up_limits_[0];

      int chunk_size = n0_total / mpi_world_size_;
      int remainder = n0_total % mpi_world_size_;

      int n0_local_start_index = mpi_rank_ * chunk_size + std::min(mpi_rank_, remainder);
      int n0_local_count = chunk_size + (mpi_rank_ < remainder ? 1 : 0);

      if (n0_local_count > 0) {
        double h0_global_step = (b0_global - a0_global) / n0_total;
        double a0_local = a0_global + n0_local_start_index * h0_global_step;
        double b0_local = a0_global + (n0_local_start_index + n0_local_count) * h0_global_step;

        if (mpi_rank_ == mpi_world_size_ - 1) {
          b0_local = b0_global;
        }
        a0_local = std::max(a0_global, a0_local);
        b0_local = std::min(b0_global, b0_local);

        if (a0_local < b0_local) {
          if (dimensions_ == 1) {
            local_integral_sum = ComputeOneDimensionalOMP(func_, a0_local, b0_local, n0_local_count);
          } else {
            local_integral_sum = ComputeOuterParallelInnerSequential(func_, a0_local, b0_local, n0_local_count,
                                                                     down_limits_, up_limits_, counts_, dimensions_);
          }
        }
      }
    }
    MPI_Reduce(&local_integral_sum, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in RunImpl: " << e.what() << std::endl;
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
  double total_sum_omp = 0.0;
#pragma omp parallel
  {
    std::vector<double> point(1);
#pragma omp for schedule(static) reduction(+ : total_sum_omp)
    for (int i = 0; i < n_local; ++i) {
      point[0] = a_local + (static_cast<double>(i) + 0.5) * step;
      total_sum_omp += f(point);
    }
  }
  return total_sum_omp * step;
}

double Integral::ComputeOuterParallelInnerSequential(const std::function<double(const std::vector<double>&)>& f,
                                                     double a0_local_mpi, double b0_local_mpi, int n0_local_mpi,
                                                     const std::vector<double>& full_a,
                                                     const std::vector<double>& full_b, const std::vector<int>& full_n,
                                                     int total_dims) {
  if (n0_local_mpi <= 0 || a0_local_mpi >= b0_local_mpi) {
    return 0.0;
  }
  const double h0_local_step = (b0_local_mpi - a0_local_mpi) / n0_local_mpi;
  double outer_integral_sum_omp = 0.0;
#pragma omp parallel
  {
    std::vector<double> current_point(static_cast<size_t>(total_dims));
#pragma omp for schedule(static) reduction(+ : outer_integral_sum_omp)
    for (int i = 0; i < n0_local_mpi; ++i) {
      current_point[0] = a0_local_mpi + (static_cast<double>(i) + 0.5) * h0_local_step;
      outer_integral_sum_omp += ComputeSequentialRecursive(f, full_a, full_b, full_n, total_dims, current_point, 1);
    }
  }
  return outer_integral_sum_omp * h0_local_step;
}

double Integral::ComputeSequentialRecursive(const std::function<double(const std::vector<double>&)>& f,
                                            const std::vector<double>& a_all_dims,
                                            const std::vector<double>& b_all_dims, const std::vector<int>& n_all_dims,
                                            int total_dims, std::vector<double>& current_eval_point,
                                            int current_dim_index) {
  if (current_dim_index == total_dims) {
    return f(current_eval_point);
  }
  if (current_dim_index < 0 || static_cast<size_t>(current_dim_index) >= n_all_dims.size() ||
      static_cast<size_t>(current_dim_index) >= a_all_dims.size() ||
      static_cast<size_t>(current_dim_index) >= b_all_dims.size()) {
    throw std::out_of_range("Dimension index out of bounds in recursive call. Index: " +
                            std::to_string(current_dim_index) + ", total_dims: " + std::to_string(total_dims));
  }
  const int n_for_current_dim = n_all_dims[static_cast<size_t>(current_dim_index)];
  const double a_for_current_dim = a_all_dims[static_cast<size_t>(current_dim_index)];
  const double b_for_current_dim = b_all_dims[static_cast<size_t>(current_dim_index)];
  if (n_for_current_dim <= 0) {
    throw std::runtime_error("Non-positive interval count (n) in recursive call for dimension " +
                             std::to_string(current_dim_index));
  }
  if (a_for_current_dim >= b_for_current_dim) {
    return 0.0;
  }
  const double h_step_for_current_dim = (b_for_current_dim - a_for_current_dim) / n_for_current_dim;
  double sum_for_this_dimension = 0.0;
  for (int i = 0; i < n_for_current_dim; ++i) {
    current_eval_point[static_cast<size_t>(current_dim_index)] =
        a_for_current_dim + (static_cast<double>(i) + 0.5) * h_step_for_current_dim;
    sum_for_this_dimension += ComputeSequentialRecursive(f, a_all_dims, b_all_dims, n_all_dims, total_dims,
                                                         current_eval_point, current_dim_index + 1);
  }
  return sum_for_this_dimension * h_step_for_current_dim;
}

bool Integral::PostProcessingImpl() {
  try {
    if (mpi_rank_ == 0) {
      if (!this->taskData) {
        throw std::invalid_argument("PostProcessingImpl Error: TaskData is null on root.");
      }
      if (this->taskData->outputs.empty() || this->taskData->outputs[0] == nullptr) {
        throw std::invalid_argument("Invalid output task data or output buffer on root process.");
      }
      if (this->taskData->outputs_count.empty() || this->taskData->outputs_count[0] != sizeof(double)) {
        throw std::invalid_argument(
            "Output data size mismatch on root process. Expected " + std::to_string(sizeof(double)) + ", got " +
            (this->taskData->outputs_count.empty() ? "0" : std::to_string(this->taskData->outputs_count[0])));
      }
      auto* outputs_ptr = reinterpret_cast<double*>(this->taskData->outputs[0]);
      outputs_ptr[0] = result_;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Error in PostProcessingImpl: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

}  // namespace shurigin_s_integrals_square_mpi