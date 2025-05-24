#include "all/anufriev_d_integrals_simpson/include/ops_all.hpp"

#include <mpi.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}
}  // namespace

namespace anufriev_d_integrals_simpson_all {

double IntegralsSimpsonAll::FunctionN(const std::vector<double>& coords) const {
  switch (func_code_) {
    case 0: {
      double s = 0.0;
      for (double c : coords) {
        s += c * c;
      }
      return s;
    }
    case 1: {
      double val = 1.0;
      for (size_t i = 0; i < coords.size(); i++) {
        if (i % 2 == 0) {
          val *= std::sin(coords[i]);
        } else {
          val *= std::cos(coords[i]);
        }
      }
      return val;
    }
    default:
      return 0.0;
  }
}

bool IntegralsSimpsonAll::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int root_status = 1;

  if (rank == 0) {
    if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
      root_status = 0;
    }

    if (root_status == 1) {
      auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
      size_t in_size_bytes = task_data->inputs_count[0];
      size_t num_doubles = in_size_bytes / sizeof(double);

      if (num_doubles < 1) {
        root_status = 0;
      }

      if (root_status == 1) {
        int d_parsed = static_cast<int>(in_ptr[0]);
        if (d_parsed <= 0) {
          root_status = 0;
        } else {
          dimension_ = d_parsed;
          size_t required_elements = 1 + static_cast<size_t>(3 * dimension_) + 1;
          if (num_doubles < required_elements) {
            root_status = 0;
          }
        }
      }

      if (root_status == 1) {
        a_.resize(dimension_);
        b_.resize(dimension_);
        n_.resize(dimension_);

        int idx_ptr = 1;
        for (int i = 0; i < dimension_; i++) {
          a_[i] = in_ptr[idx_ptr++];
          b_[i] = in_ptr[idx_ptr++];
          double n_double = in_ptr[idx_ptr++];
          if (std::floor(n_double) != n_double || n_double > static_cast<double>(std::numeric_limits<int>::max()) ||
              n_double <= 0.0 || (static_cast<int>(n_double) % 2 != 0)) {
            root_status = 0;
            break;
          }
          n_[i] = static_cast<int>(n_double);
        }
        if (root_status == 1) {
          double func_code_double = in_ptr[idx_ptr];
          if (std::floor(func_code_double) != func_code_double ||
              func_code_double > static_cast<double>(std::numeric_limits<int>::max()) ||
              func_code_double < static_cast<double>(std::numeric_limits<int>::min())) {
            root_status = 0;
          } else {
            func_code_ = static_cast<int>(func_code_double);
          }
        }
      }
    }
  }

  MPI_Bcast(&root_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (root_status == 0) {
    return false;
  }

  MPI_Bcast(&dimension_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    if (dimension_ <= 0) return false;
    a_.resize(dimension_);
    b_.resize(dimension_);
    n_.resize(dimension_);
  }

  MPI_Bcast(a_.data(), dimension_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_.data(), dimension_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(n_.data(), dimension_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&func_code_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    for (int val_n : n_) {
      if (val_n <= 0 || (val_n % 2) != 0) {
        return false;
      }
    }
  }

  result_ = 0.0;
  return true;
}

bool IntegralsSimpsonAll::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int validation_status_root = 1;

  if (rank == 0) {
    if (task_data == nullptr) {
        validation_status_root = 0;
    } else if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
      validation_status_root = 0;
    } else if (task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
      validation_status_root = 0;
    }
  }

  MPI_Bcast(&validation_status_root, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return (validation_status_root == 1);
}rn true;
}

bool IntegralsSimpsonAll::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<double> steps(dimension_);
  size_t total_points = 1;
  double coeff_mult = 1.0;

  for (int i = 0; i < dimension_; i++) {
    if (n_[i] == 0) {
      return false;
    }
    steps[i] = (b_[i] - a_[i]) / n_[i];
    coeff_mult *= steps[i] / 3.0;
    size_t points_in_dim = static_cast<size_t>(n_[i]) + 1;

    if (total_points > std::numeric_limits<size_t>::max() / points_in_dim) {
      return false;
    }
    total_points *= points_in_dim;
  }

  if (total_points == 0 && dimension_ > 0) {
    result_ = 0.0;
  }

  size_t points_per_rank_base = total_points / static_cast<size_t>(world_size);
  size_t remainder_points = total_points % static_cast<size_t>(world_size);

  size_t local_start_k;
  size_t num_points_for_this_rank;

  if (static_cast<size_t>(rank) < remainder_points) {
    num_points_for_this_rank = points_per_rank_base + 1;
    local_start_k = static_cast<size_t>(rank) * num_points_for_this_rank;
  } else {
    num_points_for_this_rank = points_per_rank_base;
    local_start_k = static_cast<size_t>(rank) * points_per_rank_base + remainder_points;
  }
  size_t local_end_k = local_start_k + num_points_for_this_rank;

  if (total_points == 0) {
    local_start_k = 0;
    local_end_k = 0;
  }

  double local_sum = 0.0;
  if (local_start_k < local_end_k) {
    local_sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(local_start_k, local_end_k), 0.0,
        [&](const tbb::blocked_range<size_t>& r, double running_sum) {
          std::vector<double> coords(dimension_);
          std::vector<int> current_idx(dimension_);

          for (size_t k = r.begin(); k != r.end(); ++k) {
            double current_coeff_prod = 1.0;
            size_t current_k = k;
            for (int dim_idx = 0; dim_idx < dimension_; ++dim_idx) {
              size_t points_in_this_dim = static_cast<size_t>(n_[dim_idx]) + 1;
              size_t index_in_this_dim = current_k % points_in_this_dim;
              current_idx[dim_idx] = static_cast<int>(index_in_this_dim);
              current_k /= points_in_this_dim;

              coords[dim_idx] = a_[dim_idx] + current_idx[dim_idx] * steps[dim_idx];
              current_coeff_prod *= SimpsonCoeff(current_idx[dim_idx], n_[dim_idx]);
            }
            running_sum += current_coeff_prod * FunctionN(coords);
          }
          return running_sum;
        },
        [](double x, double y) { return x + y; });
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result_ = coeff_mult * global_sum;
  } else {
    result_ = local_sum;
  }

  return true;
}

bool IntegralsSimpsonAll::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    try {
      if (task_data->outputs.empty() || task_data->outputs[0] == nullptr || task_data->outputs_count.empty() ||
          task_data->outputs_count[0] < sizeof(double)) {
        std::cerr << "Error: Output buffer not properly set up for rank 0 during PostProcessing.\n";
        return false;
      }
      auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
      out_ptr[0] = result_;
    } catch (const std::exception& e) {
      std::cerr << "Error during PostProcessing on rank 0: " << e.what() << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace anufriev_d_integrals_simpson_all