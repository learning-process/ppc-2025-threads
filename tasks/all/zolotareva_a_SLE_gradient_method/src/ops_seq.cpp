#include "all/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <vector>

bool zolotareva_a_sle_gradient_method_all::TestTaskALL::PreProcessingImpl() {
  if (world.rank() == 0) {
    n_ = static_cast<int>(task_data->inputs_count[1]);
    const auto* input_matrix = reinterpret_cast<const double*>(task_data->inputs[0]);
    const auto* input_vector = reinterpret_cast<const double*>(task_data->inputs[1]);
    a_.resize(n_ * n_);
    b_.resize(n_);

    for (int i = 0; i < n_; ++i) {
      b_[i] = input_vector[i];
      for (int j = 0; j < n_; ++j) {
        a_[i * n_ + j] = input_matrix[i * n_ + j];
      }
    }
  }

  return true;
}

bool zolotareva_a_sle_gradient_method_all::TestTaskALL::ValidationImpl() {
  if (world.rank() == 0) {
    if (static_cast<int>(task_data->inputs_count[0]) < 0 || static_cast<int>(task_data->inputs_count[1]) < 0 ||
        static_cast<int>(task_data->outputs_count[0]) < 0) {
      return false;
    }
    if (task_data->inputs_count.size() < 2 || task_data->inputs.size() < 2 || task_data->outputs.empty()) {
      return false;
    }

    if (static_cast<int>(task_data->inputs_count[0]) !=
        (static_cast<int>(task_data->inputs_count[1]) * static_cast<int>(task_data->inputs_count[1]))) {
      return false;
    }
    if (task_data->outputs_count[0] != task_data->inputs_count[1]) {
      return false;
    }

    // проверка симметрии и положительной определённости
    const auto* a = reinterpret_cast<const double*>(task_data->inputs[0]);

    return IsPositiveAndSimm(a, static_cast<int>(task_data->inputs_count[1]));
  }
  return true;
}

bool zolotareva_a_sle_gradient_method_all::TestTaskALL::RunImpl() {
  int world_size = world.size();
  int rank = world.rank();

  boost::mpi::broadcast(world, n_, 0);

  int base_rows = n_ / world_size;
  int remainder = n_ % world_size;
  local_rows = base_rows;

  if (rank == 0) {
    local_rows += remainder;

    int start_row = local_rows;
    for (int proc = 1; proc < world_size; ++proc) {
      world.send(proc, 0, a_.data() + start_row * n_, base_rows * n_);
      world.send(proc, 1, b_.data() + start_row, base_rows);
      start_row += base_rows;
    }

    local_a_.resize(local_rows * n_);
    local_b_.resize(local_rows);
#pragma omp parallel for
    for (int i = 0; i < local_rows; ++i) {
      local_b_[i] = b_[i];
      for (int j = 0; j < n_; ++j) {
        local_a_[i * n_ + j] = a_[i * n_ + j];
      }
    }
  } else {
    local_a_.resize(local_rows * n_);
    local_b_.resize(local_rows);
    world.recv(0, 0, local_a_.data(), local_rows * n_);
    world.recv(0, 1, local_b_.data(), local_rows);
  }

  local_x_.assign(local_rows, 0.0);
  std::vector<double> r = local_b_;
  std::vector<double> p = r;

  int local_rows_0 = base_rows + remainder;
  if (world_size < 1) world size = 1;
  std::vector<int> recvcounts(world_size, base_rows);
  recvcounts[0] = local_rows_0;
  std::vector<int> displs(world_size, 0);
  for (int i = 1; i < world_size; ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  std::vector<double> global_p(n_);
  std::vector<double> Ap(local_rows);

  double rs_old = 0.0;
#pragma omp parallel for reduction(+ : rs_old)
  for (int i = 0; i < local_rows; ++i) {
    rs_old += r[i] * r[i];
  }

  double rs_global_old;
  boost::mpi::all_reduce(world, rs_old, rs_global_old, std::plus<>());
  double initial_res_norm = std::sqrt(rs_global_old);
  double threshold = (initial_res_norm == 0.0) ? 1e-12 : (1e-12 * initial_res_norm);

  for (int iter = 0; iter <= n_; ++iter) {
    MPI_Allgatherv(p.data(), local_rows, MPI_DOUBLE, global_p.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   world);

#pragma omp parallel for
    for (int i = 0; i < local_rows; ++i) {
      double sum = 0.0;
      double const* row = &local_a_[i * n_];
      for (int j = 0; j < n_; ++j) sum += row[j] * global_p[j];
      Ap[i] = sum;
    }

    double local_dot_pAp = 0.0;
#pragma omp parallel for reduction(+ : local_dot_pAp)
    for (int i = 0; i < local_rows; ++i) local_dot_pAp += p[i] * Ap[i];

    double global_dot_pAp;
    boost::mpi::all_reduce(world, local_dot_pAp, global_dot_pAp, std::plus<>());
    if (global_dot_pAp == 0.0) break;

    double alpha = rs_global_old / global_dot_pAp;

#pragma omp parallel for
    for (int i = 0; i < local_rows; ++i) {
      local_x_[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double local_rs_new = 0.0;
#pragma omp parallel for reduction(+ : local_rs_new)
    for (int i = 0; i < local_rows; ++i) local_rs_new += r[i] * r[i];

    double rs_global_new;
    boost::mpi::all_reduce(world, local_rs_new, rs_global_new, std::plus<>());
    if (rs_global_new < threshold) break;

    double beta = rs_global_new / rs_global_old;

#pragma omp parallel for
    for (int i = 0; i < local_rows; ++i) p[i] = r[i] + beta * p[i];

    rs_global_old = rs_global_new;
  }

  if (world.rank() == 0) {
    x_.resize(n_);
    std::copy(local_x_.begin(), local_x_.end(), x_.begin());
    int start_row = local_rows;

    std::vector<double> buffer(base_rows);
    for (int proc = 1; proc < world.size(); ++proc) {
      world.recv(proc, 2, buffer);
      std::copy(buffer.begin(), buffer.end(), x_.begin() + start_row);
      start_row += base_rows;
    }
  } else
    world.send(0, 2, local_x_);

  return true;
}

bool zolotareva_a_sle_gradient_method_all::TestTaskALL::PostProcessingImpl() {
  if (world.rank() == 0) {
    auto* output_raw = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(x_.begin(), x_.end(), output_raw);
  }
  return true;
}
bool zolotareva_a_sle_gradient_method_all::TestTaskALL::IsPositiveAndSimm(const double* a, int n) {
  std::vector<double> m(n * n);
  // копируем и проверяем симметричность
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double val = a[(i * n) + j];
      m[(i * n) + j] = val;
      if (j > i) {
        if (val != a[(j * n) + i]) {
          return false;
        }
      }
    }
  }
  // проверяем позитивную определенность
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = m[(i * n) + j];
      for (int k = 0; k < j; k++) {
        sum -= m[(i * n) + k] * m[(j * n) + k];
      }

      if (i == j) {
        if (sum <= 1e-15) {
          return false;
        }
        m[(i * n) + j] = std::sqrt(sum);
      } else {
        m[(i * n) + j] = sum / m[(j * n) + j];
      }
    }
  }
  return true;
}
