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

  if (world_size > 1) {
    DistributeData();
  }
  return true;
}

void SparseMatmulTask::DistributeData() {
  // Broadcast matrix A to all processes
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

  // Broadcast matrix B to all processes
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

void SparseMatmulTask::GatherResults() {
  if (world_size == 1) return;

  // Gather sizes from all processes
  std::vector<int> all_sizes(world_size);
  int local_size = C_values.size();
  MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Gather all data to root process
  if (world_rank == 0) {
    std::vector<double> all_values;
    std::vector<int> all_row_indices;
    std::vector<int> displacements(world_size + 1, 0);

    for (int i = 0; i < world_size; ++i) {
      displacements[i + 1] = displacements[i] + all_sizes[i];
    }
    all_values.resize(displacements.back());
    all_row_indices.resize(displacements.back());

    MPI_Gatherv(C_values.data(), local_size, MPI_DOUBLE, all_values.data(), all_sizes.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(C_row_indices.data(), local_size, MPI_INT, all_row_indices.data(), all_sizes.data(),
                displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // Merge results from all processes
    C_values = all_values;
    C_row_indices = all_row_indices;
  } else {
    MPI_Gatherv(C_values.data(), local_size, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(C_row_indices.data(), local_size, MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
  }
}

bool SparseMatmulTask::RunImpl() {
  int local_num_threads = ppc::util::GetPPCNumThreads();
  if (local_num_threads == 0) {
    local_num_threads = 4;
  }

  // Divide work among MPI processes
  int cols_per_proc = colsB / world_size;
  int remainder = colsB % world_size;
  int start_col = world_rank * cols_per_proc + std::min(world_rank, remainder);
  int end_col = start_col + cols_per_proc + (world_rank < remainder ? 1 : 0);

  std::vector<std::vector<double>> thread_c_values(local_num_threads);
  std::vector<std::vector<int>> thread_c_row_indices(local_num_threads);
  std::vector<std::vector<int>> thread_c_col_ptr(local_num_threads, std::vector<int>(colsB + 1, 0));

  auto worker = [&](int thread_id) {
    for (int col_b = start_col + thread_id; col_b < end_col; col_b += local_num_threads) {
      ProcessColumn(thread_id, col_b, thread_c_values[thread_id], thread_c_row_indices[thread_id],
                    thread_c_col_ptr[thread_id]);
    }

    for (int col = 1; col <= colsB; ++col) {
      thread_c_col_ptr[thread_id][col] += thread_c_col_ptr[thread_id][col - 1];
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(local_num_threads);
  for (int t = 0; t < local_num_threads; ++t) {
    threads.emplace_back(worker, t);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  C_values.clear();
  C_row_indices.clear();
  C_col_ptr.assign(colsB + 1, 0);

  MergeThreadResults(local_num_threads, thread_c_values, thread_c_row_indices, thread_c_col_ptr);

  if (world_size > 1) {
    GatherResults();
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all