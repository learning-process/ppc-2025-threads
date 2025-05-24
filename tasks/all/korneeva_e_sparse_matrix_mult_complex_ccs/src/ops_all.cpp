#include "all/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace korneeva_e_sparse_matrix_mult_complex_ccs_all {

bool SparseMatrixMultComplexCCS::PreProcessingImpl() {
  matrix1_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  matrix2_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);
  result_ = SparseMatrixCCS(matrix1_->rows, matrix2_->cols, 0);
  return true;
}

bool SparseMatrixMultComplexCCS::ValidationImpl() {
  if (task_data->inputs.size() != 2 || task_data->outputs.size() != 1) {
    return false;
  }

  auto* m1 = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  auto* m2 = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);

  return m1 != nullptr && m2 != nullptr && m1->cols == m2->rows && m1->rows > 0 && m1->cols > 0 && m2->rows > 0 &&
         m2->cols > 0;
}

bool SparseMatrixMultComplexCCS::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_cols = matrix2_->cols;
  int cols_per_process = total_cols / size;
  int remaining_cols = total_cols % size;

  int start_col = (rank * cols_per_process) + std::min(rank, remaining_cols);
  int extra_col = rank < remaining_cols ? 1 : 0;
  int end_col = start_col + cols_per_process + extra_col;

  std::vector<std::vector<std::pair<Complex, int>>> column_results(end_col - start_col);
  std::vector<int> col_indices(end_col - start_col);
  std::iota(col_indices.begin(), col_indices.end(), start_col);

  int num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::max(1, num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  int cols_per_thread = (end_col - start_col) / num_threads;
  int remaining_thread_cols = (end_col - start_col) % num_threads;

  auto compute_range = [&](int thread_start, int thread_end) {
    for (int j = thread_start; j < thread_end; ++j) {
      ComputeColumn(col_indices[j], column_results[j - start_col]);
    }
  };

  int thread_start = start_col;
  for (int i = 0; i < num_threads; ++i) {
    int cols = cols_per_thread + (i < remaining_thread_cols ? 1 : 0);
    int thread_end = thread_start + cols;
    threads.emplace_back(compute_range, thread_start, thread_end);
    thread_start = thread_end;
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::vector<Complex> local_values;
  std::vector<int> local_row_indices;
  std::vector<int> local_col_offsets(end_col - start_col + 1, 0);

  int local_nnz = 0;
  for (int j = 0; j < end_col - start_col; ++j) {
    auto& col_data = column_results[j];
    for (const auto& [value, row_idx] : col_data) {
      local_values.push_back(value);
      local_row_indices.push_back(row_idx);
      local_nnz++;
    }
    local_col_offsets[j + 1] = local_nnz;
  }

  std::vector<int> all_nnz(size);
  MPI_Allgather(&local_nnz, 1, MPI_INT, all_nnz.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int total_nnz = std::accumulate(all_nnz.begin(), all_nnz.end(), 0);
  std::vector<int> displacements(size);
  if (!displacements.empty()) {
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
      displacements[i] = displacements[i - 1] + all_nnz[i - 1];
    }
  }
  result_.values.resize(total_nnz);
  result_.row_indices.resize(total_nnz);
  result_.col_offsets.resize(total_cols + 1, 0);
  result_.nnz = total_nnz;

  MPI_Allgatherv(local_values.data(), local_nnz, MPI_CXX_DOUBLE_COMPLEX, result_.values.data(), all_nnz.data(),
                 displacements.data(), MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
  MPI_Allgatherv(local_row_indices.data(), local_nnz, MPI_INT, result_.row_indices.data(), all_nnz.data(),
                 displacements.data(), MPI_INT, MPI_COMM_WORLD);

  std::vector<int> local_col_counts(total_cols, 0);
  for (int j = 0; j < end_col - start_col; ++j) {
    local_col_counts[start_col + j] = static_cast<int>(column_results[j].size());
  }
  std::vector<int> global_col_counts(total_cols);
  MPI_Reduce(local_col_counts.data(), global_col_counts.data(), total_cols, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result_.col_offsets[0] = 0;
    for (int j = 0; j < total_cols; ++j) {
      result_.col_offsets[j + 1] = result_.col_offsets[j] + global_col_counts[j];
    }
  }
  MPI_Bcast(result_.col_offsets.data(), total_cols + 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void SparseMatrixMultComplexCCS::ComputeColumn(int col_idx, std::vector<std::pair<Complex, int>>& column_data) {
  int col_start2 = matrix2_->col_offsets[col_idx];
  int col_end2 = matrix2_->col_offsets[col_idx + 1];

  column_data.reserve(matrix1_->rows);
  for (int i = 0; i < matrix1_->rows; i++) {
    Complex sum = ComputeElement(i, col_start2, col_end2);
    if (sum != Complex(0.0, 0.0)) {
      column_data.emplace_back(sum, i);
    }
  }
}

Complex SparseMatrixMultComplexCCS::ComputeElement(int row_idx, int col_start2, int col_end2) {
  Complex sum(0.0, 0.0);
  for (int k = 0; k < matrix1_->cols; k++) {
    int col_start1 = matrix1_->col_offsets[k];
    int col_end1 = matrix1_->col_offsets[k + 1];
    sum += ComputeContribution(row_idx, k, col_start1, col_end1, col_start2, col_end2);
  }
  return sum;
}

Complex SparseMatrixMultComplexCCS::ComputeContribution(int row_idx, int k, int col_start1, int col_end1,
                                                        int col_start2, int col_end2) {
  Complex contribution(0.0, 0.0);
  for (int p = col_start1; p < col_end1; p++) {
    if (matrix1_->row_indices[p] == row_idx) {
      for (int q = col_start2; q < col_end2; q++) {
        if (matrix2_->row_indices[q] == k) {
          contribution += matrix1_->values[p] * matrix2_->values[q];
        }
      }
    }
  }
  return contribution;
}

bool SparseMatrixMultComplexCCS::PostProcessingImpl() {
  *reinterpret_cast<SparseMatrixCCS*>(task_data->outputs[0]) = result_;
  return true;
}

}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_all