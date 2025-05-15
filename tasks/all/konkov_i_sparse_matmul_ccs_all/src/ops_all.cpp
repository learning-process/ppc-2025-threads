#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) return false;
  if (A_col_ptr.empty() || B_col_ptr.empty()) return false;
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() {
  C_col_ptr.clear();
  C_row_indices.clear();
  C_values.clear();

  boost::mpi::broadcast(world, A_values, 0);
  boost::mpi::broadcast(world, A_row_indices, 0);
  boost::mpi::broadcast(world, A_col_ptr, 0);
  boost::mpi::broadcast(world, rowsA, 0);
  boost::mpi::broadcast(world, colsA, 0);

  boost::mpi::broadcast(world, B_values, 0);
  boost::mpi::broadcast(world, B_row_indices, 0);
  boost::mpi::broadcast(world, B_col_ptr, 0);
  boost::mpi::broadcast(world, rowsB, 0);
  boost::mpi::broadcast(world, colsB, 0);

  return true;
}

void SparseMatmulTask::ProcessColumn(int col_b, int start_col, std::vector<double>& local_values,
                                     std::vector<int>& local_rows, std::vector<int>& local_col_ptr) {
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

  std::vector<std::pair<int, double>> sorted_entries;
  for (const auto& [row, val] : column_result) {
    if (val != 0.0) sorted_entries.emplace_back(row, val);
  }
  std::sort(sorted_entries.begin(), sorted_entries.end());

  int local_col = col_b - start_col;
  local_col_ptr[local_col + 1] = static_cast<int>(sorted_entries.size());

  for (const auto& [row, val] : sorted_entries) {
    local_rows.push_back(row);
    local_values.push_back(val);
  }
}

bool SparseMatmulTask::RunImpl() {
  int rank = world.rank();
  int size = world.size();

  int base_cols = colsB / size;
  int extra_cols = colsB % size;
  int start_col = rank * base_cols + std::min(rank, extra_cols);
  int end_col = start_col + base_cols + (rank < extra_cols ? 1 : 0);
  int num_local_cols = end_col - start_col;

  std::vector<double> local_values;
  std::vector<int> local_rows;
  std::vector<int> local_col_ptr(num_local_cols + 1, 0);

  int num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads <= 0) num_threads = 4;

  std::vector<std::vector<double>> thread_values(num_threads);
  std::vector<std::vector<int>> thread_rows(num_threads);
  std::vector<std::vector<int>> thread_col_ptrs(num_threads, std::vector<int>(num_local_cols + 1, 0));

  auto worker = [&](int thread_id) {
    for (int col = start_col + thread_id; col < end_col; col += num_threads) {
      ProcessColumn(col, start_col, thread_values[thread_id], thread_rows[thread_id], thread_col_ptrs[thread_id]);
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) t.join();

  for (int local_col = 0; local_col < num_local_cols; ++local_col) {
    for (int t = 0; t < num_threads; ++t) {
      local_col_ptr[local_col + 1] += thread_col_ptrs[t][local_col + 1];
    }
  }

  for (int col = 1; col <= num_local_cols; ++col) {
    local_col_ptr[col] += local_col_ptr[col - 1];
  }

  for (int t = 0; t < num_threads; ++t) {
    local_values.insert(local_values.end(), thread_values[t].begin(), thread_values[t].end());
    local_rows.insert(local_rows.end(), thread_rows[t].begin(), thread_rows[t].end());
  }

  std::vector<int> proc_start_cols(size), proc_end_cols(size);
  proc_start_cols[rank] = start_col;
  proc_end_cols[rank] = end_col;

  boost::mpi::gather(world, start_col, proc_start_cols, 0);
  boost::mpi::gather(world, end_col, proc_end_cols, 0);

  std::vector<std::vector<double>> all_values;
  std::vector<std::vector<int>> all_rows;
  std::vector<std::vector<int>> all_col_ptrs;

  boost::mpi::gather(world, local_values, all_values, 0);
  boost::mpi::gather(world, local_rows, all_rows, 0);
  boost::mpi::gather(world, local_col_ptr, all_col_ptrs, 0);

  if (rank == 0) {
    C_values.clear();
    C_row_indices.clear();
    C_col_ptr.resize(colsB + 1, 0);

    for (int global_col = 0; global_col < colsB; ++global_col) {
      for (int proc = 0; proc < size; ++proc) {
        if (global_col >= proc_start_cols[proc] && global_col < proc_end_cols[proc]) {
          int local_col = global_col - proc_start_cols[proc];
          int start = all_col_ptrs[proc][local_col];
          int end = all_col_ptrs[proc][local_col + 1];

          for (int i = start; i < end; ++i) {
            C_values.push_back(all_values[proc][i]);
            C_row_indices.push_back(all_rows[proc][i]);
          }
          C_col_ptr[global_col + 1] = C_col_ptr[global_col] + (end - start);
          break;
        }
      }
    }
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_all