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
  std::vector<std::unordered_map<int, double>> column_results(colsB);

  for (int t = 0; t < num_threads; ++t) {
    int offset = 0;
    for (int col = 0; col < colsB; ++col) {
      int start = thread_c_col_ptr[t][col];
      int end = thread_c_col_ptr[t][col + 1];

      for (int i = start; i < end; ++i) {
        int row = thread_c_row_indices[t][i];
        double val = thread_c_values[t][i];
        column_results[col][row] += val;
      }
    }
  }

  C_col_ptr.resize(colsB + 1, 0);
  C_values.clear();
  C_row_indices.clear();

  for (int col = 0; col < colsB; ++col) {
    C_col_ptr[col + 1] = C_col_ptr[col];

    std::vector<int> rows;
    for (const auto& pair : column_results[col]) {
      if (pair.second != 0.0) {
        rows.push_back(pair.first);
      }
    }
    std::sort(rows.begin(), rows.end());

    for (int row : rows) {
      C_values.push_back(column_results[col][row]);
      C_row_indices.push_back(row);
      C_col_ptr[col + 1]++;
    }
  }
}

bool SparseMatmulTask::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 4;  // fallback
  }

  std::vector<std::vector<double>> thread_c_values(num_threads);
  std::vector<std::vector<int>> thread_c_row_indices(num_threads);
  std::vector<std::vector<int>> thread_c_col_ptr(num_threads, std::vector<int>(colsB + 1, 0));

  auto worker = [&](int thread_id) {
    for (int col_b = thread_id + rank; col_b < colsB; col_b += num_threads * size) {
      ProcessColumn(thread_id, col_b, thread_c_values[thread_id], thread_c_row_indices[thread_id],
                    thread_c_col_ptr[thread_id]);
    }

    for (int col = 1; col <= colsB; ++col) {
      thread_c_col_ptr[thread_id][col] += thread_c_col_ptr[thread_id][col - 1];
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  if (rank == 0) {
    std::vector<std::unordered_map<int, double>> global_column_results(colsB);

    for (int col = 0; col < colsB; ++col) {
      for (int i = C_col_ptr[col]; i < C_col_ptr[col + 1]; ++i) {
        global_column_results[col][C_row_indices[i]] += C_values[i];
      }
    }

    for (int src = 1; src < size; ++src) {
      std::vector<int> recv_col_ptr(colsB + 1);
      MPI_Recv(recv_col_ptr.data(), colsB + 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int values_size = recv_col_ptr.back();
      std::vector<double> recv_values(values_size);
      std::vector<int> recv_row_indices(values_size);

      MPI_Recv(recv_values.data(), values_size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(recv_row_indices.data(), values_size, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int col = 0; col < colsB; ++col) {
        for (int i = recv_col_ptr[col]; i < recv_col_ptr[col + 1]; ++i) {
          global_column_results[col][recv_row_indices[i]] += recv_values[i];
        }
      }
    }

    C_col_ptr.resize(colsB + 1, 0);
    C_values.clear();
    C_row_indices.clear();

    for (int col = 0; col < colsB; ++col) {
      C_col_ptr[col + 1] = C_col_ptr[col];

      std::vector<int> rows;
      for (const auto& pair : global_column_results[col]) {
        if (pair.second != 0.0) {
          rows.push_back(pair.first);
        }
      }
      std::sort(rows.begin(), rows.end());

      for (int row : rows) {
        C_values.push_back(global_column_results[col][row]);
        C_row_indices.push_back(row);
        C_col_ptr[col + 1]++;
      }
    }
  } else {
    // Send results to rank 0
    std::vector<double> send_values;
    std::vector<int> send_row_indices, send_col_ptr(colsB + 1, 0);

    for (int t = 0; t < num_threads; ++t) {
      send_values.insert(send_values.end(), thread_c_values[t].begin(), thread_c_values[t].end());
      send_row_indices.insert(send_row_indices.end(), thread_c_row_indices[t].begin(), thread_c_row_indices[t].end());
      for (int col = 0; col <= colsB; ++col) {
        send_col_ptr[col] += thread_c_col_ptr[t][col];
      }
    }

    MPI_Send(send_col_ptr.data(), colsB + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(send_values.data(), send_values.size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    MPI_Send(send_row_indices.data(), send_row_indices.size(), MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all