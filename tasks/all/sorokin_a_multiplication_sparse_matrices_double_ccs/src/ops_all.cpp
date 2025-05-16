#include "all/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_all {

void DistributeBColumns(boost::mpi::communicator& world, int rank, int size, const std::vector<double>& b_values,
                        const std::vector<int>& b_row_indices, const std::vector<int>& b_col_ptr, int n,
                        int base_cols_per_proc, int remainder, std::vector<double>& local_b_values,
                        std::vector<int>& local_b_row_indices, std::vector<int>& local_b_col_ptr) {
  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      const int p_start = (p * base_cols_per_proc) + std::min(p, remainder);
      const int p_num_cols = base_cols_per_proc + (p < remainder ? 1 : 0);
      const int b_start_idx = b_col_ptr[p_start];
      const int b_end_idx = b_col_ptr[p_start + p_num_cols];

      std::vector<double> p_b_values(b_values.begin() + b_start_idx, b_values.begin() + b_end_idx);
      std::vector<int> p_b_row_indices(b_row_indices.begin() + b_start_idx, b_row_indices.begin() + b_end_idx);
      std::vector<int> p_b_col_ptr(p_num_cols + 1);

      for (int i = 0; i <= p_num_cols; ++i) {
        p_b_col_ptr[i] = b_col_ptr[p_start + i] - b_start_idx;
      }

      if (p == 0) {
        local_b_values.swap(p_b_values);
        local_b_row_indices.swap(p_b_row_indices);
        local_b_col_ptr.swap(p_b_col_ptr);
      } else {
        world.send(p, 0, p_b_values);
        world.send(p, 1, p_b_row_indices);
        world.send(p, 2, p_b_col_ptr);
      }
    }
  } else {
    world.recv(0, 0, local_b_values);
    world.recv(0, 1, local_b_row_indices);
    world.recv(0, 2, local_b_col_ptr);
  }
}

void CalculateLocalNNZ(const std::vector<int>& a_row_indices, const std::vector<int>& a_col_ptr,
                       const std::vector<int>& local_b_row_indices, const std::vector<int>& local_b_col_ptr, int m,
                       int num_local_cols, std::vector<int>& local_nnz) {
#pragma omp parallel for
  for (int j = 0; j < num_local_cols; ++j) {
    std::vector<bool> used(m, false);
    for (int t = local_b_col_ptr[j]; t < local_b_col_ptr[j + 1]; ++t) {
      const int row_b = local_b_row_indices[t];
      for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
        const int row_a = a_row_indices[i];
        if (!used[row_a]) {
          used[row_a] = true;
          local_nnz[j]++;
        }
      }
    }
  }
}

void ComputeLocalC(const std::vector<double>& a_values, const std::vector<int>& a_row_indices,
                   const std::vector<int>& a_col_ptr, const std::vector<double>& local_b_values,
                   const std::vector<int>& local_b_row_indices, const std::vector<int>& local_b_col_ptr, int m,
                   int num_local_cols, const std::vector<int>& local_nnz, std::vector<double>& local_c_values,
                   std::vector<int>& local_c_row_indices, std::vector<int>& local_c_col_ptr) {
  local_c_col_ptr.resize(num_local_cols + 1, 0);
  for (int j = 0; j < num_local_cols; ++j) {
    local_c_col_ptr[j + 1] = local_c_col_ptr[j] + local_nnz[j];
  }

  local_c_values.resize(local_c_col_ptr.back());
  local_c_row_indices.resize(local_c_col_ptr.back());

#pragma omp parallel for
  for (int j = 0; j < num_local_cols; ++j) {
    std::vector<double> tmp(m, 0.0);
    std::vector<bool> used(m, false);
    int pos = local_c_col_ptr[j];

    for (int t = local_b_col_ptr[j]; t < local_b_col_ptr[j + 1]; ++t) {
      const int row_b = local_b_row_indices[t];
      const double val_b = local_b_values[t];
      for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
        const int row_a = a_row_indices[i];
        tmp[row_a] += a_values[i] * val_b;
        if (!used[row_a]) {
          used[row_a] = true;
        }
      }
    }

    for (int row = 0; row < m; ++row) {
      if (used[row] && tmp[row] != 0) {
        local_c_row_indices[pos] = row;
        local_c_values[pos] = tmp[row];
        pos++;
      }
    }
  }
}

void GatherResults(boost::mpi::communicator& world, int rank, int size, int n, int base_cols_per_proc, int remainder,
                   int num_local_cols, int start_col, const std::vector<int>& local_nnz,
                   const std::vector<double>& local_c_values, const std::vector<int>& local_c_row_indices,
                   const std::vector<int>& local_c_col_ptr, std::vector<double>& c_values,
                   std::vector<int>& c_row_indices, std::vector<int>& c_col_ptr) {
  if (rank == 0) {
    c_col_ptr.resize(n + 1);
    std::vector<int> gather_nnz(n, 0);

    for (int p = 0; p < size; ++p) {
      const int p_start = (p * base_cols_per_proc) + std::min(p, remainder);
      const int p_num_cols = base_cols_per_proc + (p < remainder ? 1 : 0);
      std::vector<int> p_nnz(p_num_cols);

      if (p == 0) {
        for (int j = 0; j < p_num_cols; ++j) {
          gather_nnz[p_start + j] = local_nnz[j];
        }
      } else {
        world.recv(p, 3, p_nnz);
        for (int j = 0; j < p_num_cols; ++j) {
          gather_nnz[p_start + j] = p_nnz[j];
        }
      }
    }

    c_col_ptr[0] = 0;
    for (int j = 0; j < n; ++j) {
      c_col_ptr[j + 1] = c_col_ptr[j] + gather_nnz[j];
    }

    c_values.resize(c_col_ptr.back());
    c_row_indices.resize(c_col_ptr.back());

    for (int p = 0; p < size; ++p) {
      const int p_start = (p * base_cols_per_proc) + std::min(p, remainder);
      const int p_num_cols = base_cols_per_proc + (p < remainder ? 1 : 0);

      if (p == 0) {
        for (int j = 0; j < p_num_cols; ++j) {
          const int global_j = p_start + j;
          const int start = c_col_ptr[global_j];
          const int count = gather_nnz[global_j];
          std::copy(local_c_values.begin() + local_c_col_ptr[j], local_c_values.begin() + local_c_col_ptr[j] + count,
                    c_values.begin() + start);
          std::copy(local_c_row_indices.begin() + local_c_col_ptr[j],
                    local_c_row_indices.begin() + local_c_col_ptr[j] + count, c_row_indices.begin() + start);
        }
      } else {
        for (int j = 0; j < p_num_cols; ++j) {
          const int global_j = p_start + j;
          const int start = c_col_ptr[global_j];
          const int count = gather_nnz[global_j];
          world.recv(p, global_j, c_row_indices.data() + start, count);
          world.recv(p, global_j, c_values.data() + start, count);
        }
      }
    }
  } else {
    world.send(0, 3, local_nnz);
    for (int j = 0; j < num_local_cols; ++j) {
      const int global_j = start_col + j;
      const int start = local_c_col_ptr[j];
      const int count = local_nnz[j];
      world.send(0, global_j, local_c_row_indices.data() + start, count);
      world.send(0, global_j, local_c_values.data() + start, count);
    }
  }
}

void MultiplyCCS(boost::mpi::communicator& world, const std::vector<double>& a_values,
                 const std::vector<int>& a_row_indices, int m, const std::vector<int>& a_col_ptr,
                 const std::vector<double>& b_values, const std::vector<int>& b_row_indices, int k,
                 const std::vector<int>& b_col_ptr, std::vector<double>& c_values, std::vector<int>& c_row_indices,
                 int n, std::vector<int>& c_col_ptr) {
  const int rank = world.rank();
  const int size = world.size();
  world.barrier();

  const int base_cols_per_proc = n / size;
  const int remainder = n % size;
  const int start_col = rank * base_cols_per_proc + std::min(rank, remainder);
  const int num_local_cols = base_cols_per_proc + (rank < remainder ? 1 : 0);

  std::vector<double> local_b_values;
  std::vector<int> local_b_row_indices;
  std::vector<int> local_b_col_ptr(num_local_cols + 1);

  DistributeBColumns(world, rank, size, b_values, b_row_indices, b_col_ptr, n, base_cols_per_proc, remainder,
                     local_b_values, local_b_row_indices, local_b_col_ptr);
  world.barrier();

  std::vector<int> local_nnz(num_local_cols, 0);
  CalculateLocalNNZ(a_row_indices, a_col_ptr, local_b_row_indices, local_b_col_ptr, m, num_local_cols, local_nnz);

  std::vector<double> local_c_values;
  std::vector<int> local_c_row_indices;
  std::vector<int> local_c_col_ptr;

  ComputeLocalC(a_values, a_row_indices, a_col_ptr, local_b_values, local_b_row_indices, local_b_col_ptr, m,
                num_local_cols, local_nnz, local_c_values, local_c_row_indices, local_c_col_ptr);

  GatherResults(world, rank, size, n, base_cols_per_proc, remainder, num_local_cols, start_col, local_nnz,
                local_c_values, local_c_row_indices, local_c_col_ptr, c_values, c_row_indices, c_col_ptr);
  world.barrier();
}

}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_all

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::PreProcessingImpl() {
  // Init value for input and output
  M_ = static_cast<int>(task_data->inputs_count[0]);
  K_ = static_cast<int>(task_data->inputs_count[1]);
  N_ = static_cast<int>(task_data->inputs_count[2]);
  auto* current_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  A_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[3]);
  current_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  std::vector<double> a_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[4]);
  A_row_indices_.resize(a_row_indices_d.size());
  std::ranges::transform(a_row_indices_d.begin(), a_row_indices_d.end(), A_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  std::vector<double> a_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[5]);
  A_col_ptr_.resize(a_col_ptr_d.size());
  std::ranges::transform(a_col_ptr_d.begin(), a_col_ptr_d.end(), A_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
  B_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[6]);
  current_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  std::vector<double> b_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[7]);
  B_row_indices_.resize(b_row_indices_d.size());
  std::ranges::transform(b_row_indices_d.begin(), b_row_indices_d.end(), B_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[5]);
  std::vector<double> b_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[8]);
  B_col_ptr_.resize(b_col_ptr_d.size());
  std::ranges::transform(b_col_ptr_d.begin(), b_col_ptr_d.end(), B_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::RunImpl() {
  MultiplyCCS(world_, A_values_, A_row_indices_, M_, A_col_ptr_, B_values_, B_row_indices_, K_, B_col_ptr_, C_values_,
              C_row_indices_, N_, C_col_ptr_);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::PostProcessingImpl() {
  std::vector<double> c_row_indices_d(C_row_indices_.size());
  std::vector<double> c_col_ptr_d(C_col_ptr_.size());
  std::ranges::transform(C_row_indices_.begin(), C_row_indices_.end(), c_row_indices_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  std::ranges::transform(C_col_ptr_.begin(), C_col_ptr_.end(), c_col_ptr_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  for (size_t i = 0; i < C_values_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = C_values_[i];
  }
  for (size_t i = 0; i < c_row_indices_d.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[1])[i] = c_row_indices_d[i];
  }
  for (size_t i = 0; i < c_col_ptr_d.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[2])[i] = c_col_ptr_d[i];
  }
  return true;
}
