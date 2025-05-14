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

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    C_col_ptr.resize(colsB + 1, 0);
    C_row_indices.clear();
    C_values.clear();
  }
  return true;
}

void SparseMatmulTask::ProcessColumn(int thread_id, int col_b, std::vector<double>& thread_values,
                                     std::vector<int>& thread_row_indices, std::vector<int>& thread_col_ptr) {
  std::unordered_map<int, double> column_result;

  for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
    int row_b = B_row_indices[j];
    double val_b = B_values[j];

    if (row_b >= colsA) {
      continue;
    }

    for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
      if (static_cast<size_t>(k) >= A_row_indices.size()) {
        continue;
      }

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

  std::ranges::sort(rows);

  for (int row : rows) {
    thread_values.push_back(column_result[row]);
    thread_row_indices.push_back(row);
    thread_col_ptr[col_b + 1]++;
  }
}

void SparseMatmulTask::MergeThreadResults(int num_threads, const std::vector<std::vector<double>>& thread_c_values,
                                          const std::vector<std::vector<int>>& thread_c_row_indices,
                                          const std::vector<std::vector<int>>& thread_c_col_ptr) {
  for (int col = 0; col < colsB; ++col) {
    for (int t = 0; t < num_threads; ++t) {
      int start = (col == 0) ? 0 : thread_c_col_ptr[t][col];
      int end = thread_c_col_ptr[t][col + 1];

      C_col_ptr[col + 1] += end - start;
      C_values.insert(C_values.end(), thread_c_values[t].begin() + start, thread_c_values[t].begin() + end);
      C_row_indices.insert(C_row_indices.end(), thread_c_row_indices[t].begin() + start,
                           thread_c_row_indices[t].begin() + end);
    }
  }

  for (int col = 1; col <= colsB; ++col) {
    C_col_ptr[col] += C_col_ptr[col - 1];
  }
}

bool SparseMatmulTask::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> process_values;
  std::vector<int> process_row_indices;
  std::vector<int> process_col_ptr(colsB + 1, 0);

  for (int col_b = rank; col_b < colsB; col_b += size) {
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

    std::vector<std::pair<int, double>> sorted_elements(column_result.begin(), column_result.end());
    std::sort(sorted_elements.begin(), sorted_elements.end());

    for (const auto& [row, val] : sorted_elements) {
      process_values.push_back(val);
      process_row_indices.push_back(row);
    }
    process_col_ptr[col_b + 1] = process_values.size();
  }

  if (rank == 0) {
    C_values = std::move(process_values);
    C_row_indices = std::move(process_row_indices);
    C_col_ptr = std::move(process_col_ptr);

    for (int src = 1; src < size; ++src) {
      std::vector<int> recv_col_ptr(colsB + 1);
      MPI_Recv(recv_col_ptr.data(), colsB + 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int values_size = recv_col_ptr.back();
      std::vector<double> recv_values(values_size);
      std::vector<int> recv_row_indices(values_size);

      MPI_Recv(recv_values.data(), values_size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(recv_row_indices.data(), values_size, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int col = 0; col < colsB; ++col) {
        if (recv_col_ptr[col] != recv_col_ptr[col + 1]) {
          int insert_pos = C_col_ptr[col + 1];

          C_values.insert(C_values.begin() + insert_pos, recv_values.begin() + recv_col_ptr[col],
                          recv_values.begin() + recv_col_ptr[col + 1]);
          C_row_indices.insert(C_row_indices.begin() + insert_pos, recv_row_indices.begin() + recv_col_ptr[col],
                               recv_row_indices.begin() + recv_col_ptr[col + 1]);

          for (int c = col + 1; c <= colsB; ++c) {
            C_col_ptr[c] += (recv_col_ptr[col + 1] - recv_col_ptr[col]);
          }
        }
      }
    }
  } else {
    MPI_Send(process_col_ptr.data(), colsB + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(process_values.data(), process_values.size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    MPI_Send(process_row_indices.data(), process_values.size(), MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all