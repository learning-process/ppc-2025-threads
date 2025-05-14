#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

bool SparseMatmulTask::ValidationImpl() {
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
    return false;
  }
  if (A_col_ptr.empty() || B_col_ptr.empty()) {
    return false;
  }
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() {
  C_col_ptr.resize(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();

  // Broadcast matrix data to all processes
  if (world_size > 1) {
    // Broadcast matrix A
    int a_size = A_values.size();
    MPI_Bcast(&a_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
      A_values.resize(a_size);
      A_row_indices.resize(a_size);
      A_col_ptr.resize(colsA + 1);
    }

    MPI_Bcast(A_values.data(), a_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_row_indices.data(), a_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(A_col_ptr.data(), colsA + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast matrix B
    int b_size = B_values.size();
    MPI_Bcast(&b_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
      B_values.resize(b_size);
      B_row_indices.resize(b_size);
      B_col_ptr.resize(colsB + 1);
    }

    MPI_Bcast(B_values.data(), b_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_row_indices.data(), b_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_col_ptr.data(), colsB + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  return true;
}

void SparseMatmulTask::ProcessColumns(int start_col, int end_col) {
  for (int col_b = start_col; col_b < end_col; ++col_b) {
    std::unordered_map<int, double> column_result;

    for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
      int row_b = B_row_indices[j];
      double val_b = B_values[j];

      for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
        int row_a = A_row_indices[k];
        double val_a = A_values[k];
        column_result[row_a] += val_a * val_b;
      }
    }

    std::vector<int> rows;
    for (const auto& pair : column_result) {
      if (pair.second != 0.0) {
        rows.push_back(pair.first);
      }
    }

    std::sort(rows.begin(), rows.end());

    C_col_ptr[col_b + 1] = C_col_ptr[col_b] + rows.size();
    for (int row : rows) {
      C_values.push_back(column_result[row]);
      C_row_indices.push_back(row);
    }
  }
}

void SparseMatmulTask::GatherResults() {
  if (world_size == 1) return;

  // Gather the number of non-zero elements per column from all processes
  std::vector<int> all_col_counts(colsB * world_size, 0);
  std::vector<int> local_col_counts(colsB, 0);

  for (int col = 0; col < colsB; ++col) {
    local_col_counts[col] = C_col_ptr[col + 1] - C_col_ptr[col];
  }

  MPI_Allgather(local_col_counts.data(), colsB, MPI_INT, all_col_counts.data(), colsB, MPI_INT, MPI_COMM_WORLD);

  // Calculate displacements for each process
  std::vector<int> displacements(world_size + 1, 0);
  for (int proc = 0; proc < world_size; ++proc) {
    for (int col = 0; col < colsB; ++col) {
      displacements[proc + 1] += all_col_counts[proc * colsB + col];
    }
  }

  // Gather all values and row indices
  std::vector<double> all_values(displacements[world_size]);
  std::vector<int> all_row_indices(displacements[world_size]);

  MPI_Gatherv(C_values.data(), C_values.size(), MPI_DOUBLE, all_values.data(), &displacements[1], displacements.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Gatherv(C_row_indices.data(), C_row_indices.size(), MPI_INT, all_row_indices.data(), &displacements[1],
              displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

  // On root process, merge the results
  if (world_rank == 0) {
    std::vector<double> merged_values;
    std::vector<int> merged_row_indices;
    std::vector<int> merged_col_ptr(colsB + 1, 0);

    for (int col = 0; col < colsB; ++col) {
      int total_nnz = 0;
      for (int proc = 0; proc < world_size; ++proc) {
        total_nnz += all_col_counts[proc * colsB + col];
      }
      merged_col_ptr[col + 1] = merged_col_ptr[col] + total_nnz;

      for (int proc = 0; proc < world_size; ++proc) {
        int proc_nnz = all_col_counts[proc * colsB + col];
        if (proc_nnz > 0) {
          int offset = proc * colsB + col;
          int start = (offset == 0) ? 0 : displacements[offset];
          int end = start + proc_nnz;

          for (int i = start; i < end; ++i) {
            merged_values.push_back(all_values[i]);
            merged_row_indices.push_back(all_row_indices[i]);
          }
        }
      }
    }

    C_values = merged_values;
    C_row_indices = merged_row_indices;
    C_col_ptr = merged_col_ptr;
  }
}

bool SparseMatmulTask::RunImpl() {
  // Divide columns among processes
  int cols_per_proc = colsB / world_size;
  int remainder = colsB % world_size;
  int start_col = world_rank * cols_per_proc + std::min(world_rank, remainder);
  int end_col = start_col + cols_per_proc + (world_rank < remainder ? 1 : 0);

  // Process assigned columns
  ProcessColumns(start_col, end_col);

  // Gather results to root process
  if (world_size > 1) {
    GatherResults();
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all