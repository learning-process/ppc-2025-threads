#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data)
    : ppc::core::Task(std::move(task_data)), mpi_initialized(false) {
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
    mpi_initialized = true;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

SparseMatmulTask::~SparseMatmulTask() {
  if (mpi_initialized) {
    MPI_Finalize();
  }
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
  C_col_ptr.assign(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();

  // Broadcast matrix dimensions first
  MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

  // First gather the column counts from all processes
  std::vector<int> local_col_counts(colsB, 0);
  for (int col = 0; col < colsB; ++col) {
    local_col_counts[col] = C_col_ptr[col + 1] - C_col_ptr[col];
  }

  std::vector<int> all_col_counts(colsB * world_size);
  MPI_Allgather(local_col_counts.data(), colsB, MPI_INT, all_col_counts.data(), colsB, MPI_INT, MPI_COMM_WORLD);

  // Calculate displacements for each column
  std::vector<int> col_offsets(colsB + 1, 0);
  for (int col = 0; col < colsB; ++col) {
    int total = 0;
    for (int proc = 0; proc < world_size; ++proc) {
      total += all_col_counts[proc * colsB + col];
    }
    col_offsets[col + 1] = col_offsets[col] + total;
  }

  // Prepare to gather all values and row indices
  std::vector<double> gathered_values(col_offsets[colsB]);
  std::vector<int> gathered_row_indices(col_offsets[colsB]);

  // For each column, gather from all processes
  for (int col = 0; col < colsB; ++col) {
    // Calculate the size each process contributes to this column
    std::vector<int> recv_counts(world_size);
    std::vector<int> displs(world_size + 1, 0);

    for (int proc = 0; proc < world_size; ++proc) {
      recv_counts[proc] = all_col_counts[proc * colsB + col];
      displs[proc + 1] = displs[proc] + recv_counts[proc];
    }

    // Gather values for this column
    double* val_ptr = C_values.data() + C_col_ptr[col];
    double* gathered_val_ptr = gathered_values.data() + col_offsets[col];
    MPI_Allgatherv(val_ptr, local_col_counts[col], MPI_DOUBLE, gathered_val_ptr, recv_counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Gather row indices for this column
    int* row_ptr = C_row_indices.data() + C_col_ptr[col];
    int* gathered_row_ptr = gathered_row_indices.data() + col_offsets[col];
    MPI_Allgatherv(row_ptr, local_col_counts[col], MPI_INT, gathered_row_ptr, recv_counts.data(), displs.data(),
                   MPI_INT, MPI_COMM_WORLD);
  }

  // On root process, build the final result
  if (world_rank == 0) {
    C_values = gathered_values;
    C_row_indices = gathered_row_indices;
    C_col_ptr = col_offsets;
  } else {
    C_values.clear();
    C_row_indices.clear();
    C_col_ptr.assign(colsB + 1, 0);
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