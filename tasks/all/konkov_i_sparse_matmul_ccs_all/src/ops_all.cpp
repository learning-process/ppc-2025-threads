#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return true;
  return colsA == rowsB && rowsA > 0 && colsB > 0 && !A_col_ptr.empty() && !B_col_ptr.empty();
}

bool SparseMatmulTask::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Broadcast matrix dimensions
  int a_dims[2]{rowsA, colsA};
  int b_dims[2]{rowsB, colsB};
  MPI_Bcast(a_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  rowsA = a_dims[0];
  colsA = a_dims[1];
  rowsB = b_dims[0];
  colsB = b_dims[1];

  // Broadcast matrix A
  int a_sizes[3];
  if (rank == 0) {
    a_sizes[0] = A_values.size();
    a_sizes[1] = A_row_indices.size();
    a_sizes[2] = A_col_ptr.size();
  }
  MPI_Bcast(a_sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    A_values.resize(a_sizes[0]);
    A_row_indices.resize(a_sizes[1]);
    A_col_ptr.resize(a_sizes[2]);
  }
  MPI_Bcast(A_values.data(), a_sizes[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(A_row_indices.data(), a_sizes[1], MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(A_col_ptr.data(), a_sizes[2], MPI_INT, 0, MPI_COMM_WORLD);

  // Broadcast matrix B
  int b_sizes[3];
  if (rank == 0) {
    b_sizes[0] = B_values.size();
    b_sizes[1] = B_row_indices.size();
    b_sizes[2] = B_col_ptr.size();
  }
  MPI_Bcast(b_sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    B_values.resize(b_sizes[0]);
    B_row_indices.resize(b_sizes[1]);
    B_col_ptr.resize(b_sizes[2]);
  }
  MPI_Bcast(B_values.data(), b_sizes[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_row_indices.data(), b_sizes[1], MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_col_ptr.data(), b_sizes[2], MPI_INT, 0, MPI_COMM_WORLD);

  C_col_ptr.assign(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();
  return true;
}

bool SparseMatmulTask::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> my_columns;
  for (int col_b = rank; col_b < colsB; col_b += size) {
    my_columns.push_back(col_b);
  }

  auto num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 4;
  }

  std::vector<std::vector<std::tuple<int, std::vector<double>, std::vector<int>>>> thread_results(num_threads);

  auto worker = [&](int tid) {
    for (size_t i = tid; i < my_columns.size(); i += num_threads) {
      int col_b = my_columns[i];
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
      std::vector<double> values;
      for (const auto& [row, val] : column_result) {
        if (val != 0.0) {
          rows.push_back(row);
          values.push_back(val);
        }
      }
      std::sort(rows.begin(), rows.end());

      thread_results[tid].emplace_back(col_b, values, rows);
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<std::tuple<int, std::vector<double>, std::vector<int>>> local_results;
  for (auto& res : thread_results) {
    local_results.insert(local_results.end(), res.begin(), res.end());
  }

  if (rank != 0) {
    int num_cols = local_results.size();
    MPI_Send(&num_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    for (auto& [col_b, vals, rows] : local_results) {
      int nnz = vals.size();
      MPI_Send(&col_b, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

      if (nnz > 0) {
        MPI_Send(vals.data(), nnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(rows.data(), nnz, MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    std::map<int, std::pair<std::vector<double>, std::vector<int>>> results_map;

    for (auto& [col_b, vals, rows] : local_results) {
      results_map[col_b] = {vals, rows};
    }

    for (int src = 1; src < size; ++src) {
      int num_cols;
      MPI_Recv(&num_cols, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int j = 0; j < num_cols; ++j) {
        int col_b, nnz;
        MPI_Recv(&col_b, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&nnz, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<double> vals(nnz);
        std::vector<int> rows(nnz);

        if (nnz > 0) {
          MPI_Recv(vals.data(), nnz, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(rows.data(), nnz, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        results_map[col_b] = {vals, rows};
      }
    }

    C_col_ptr.resize(colsB + 1);
    C_col_ptr[0] = 0;

    for (int col = 0; col < colsB; ++col) {
      auto& [vals, rows] = results_map[col];
      C_values.insert(C_values.end(), vals.begin(), vals.end());
      C_row_indices.insert(C_row_indices.end(), rows.begin(), rows.end());
      C_col_ptr[col + 1] = C_col_ptr[col] + vals.size();
    }
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all