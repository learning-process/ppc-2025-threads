#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  std::cout << "[Validation] Checking matrix dimensions..." << std::endl;
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
    std::cerr << "[Validation Error] colsA (" << colsA << ") != rowsB (" << rowsB << ")" << std::endl;
    return false;
  }
  if (A_col_ptr.empty() || B_col_ptr.empty()) {
    std::cerr << "[Validation Error] Empty column pointers" << std::endl;
    return false;
  }
  std::cout << "[Validation] Passed" << std::endl;
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() {
  std::cout << "\n[PreProcessing] Rank " << world.rank() << " started" << std::endl;
  C_col_ptr.clear();
  C_row_indices.clear();
  C_values.clear();

  if (world.rank() == 0) {
    std::cout << "[PreProcessing] Broadcasting matrix data from root..." << std::endl;
  }

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

  std::cout << "[PreProcessing] Rank " << world.rank() << " completed. Matrix sizes: "
            << "A(" << rowsA << "x" << colsA << "), B(" << rowsB << "x" << colsB << ")" << std::endl;
  return true;
}

void SparseMatmulTask::ProcessColumn(int col_b, int start_col, std::vector<double>& local_values,
                                     std::vector<int>& local_rows, std::vector<int>& local_col_ptr) {
  std::cout << "\n[ProcessColumn] Rank " << world.rank() << " processing column B[" << col_b
            << "] (local_col=" << (col_b - start_col) << ")" << std::endl;

  std::unordered_map<int, double> column_result;
  std::cout << "  B column range: [" << B_col_ptr[col_b] << ", " << B_col_ptr[col_b + 1] << ")" << std::endl;

  for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
    int row_b = B_row_indices[j];
    double val_b = B_values[j];
    std::cout << "  Processing B[" << row_b << "][" << col_b << "] = " << val_b << std::endl;

    std::cout << "  Multiplying with A column " << row_b << " (size=" << (A_col_ptr[row_b + 1] - A_col_ptr[row_b])
              << ")" << std::endl;

    for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
      int row_a = A_row_indices[k];
      double val_a = A_values[k];
      column_result[row_a] += val_a * val_b;
      std::cout << "    A[" << row_a << "][" << row_b << "] * " << val_b << " = " << val_a * val_b << " → C[" << row_a
                << "][" << col_b << "]" << std::endl;
    }
  }

  std::cout << "  Column result (unsorted):\n";
  for (const auto& [row, val] : column_result) {
    std::cout << "    C[" << row << "][" << col_b << "] = " << val << std::endl;
  }

  std::vector<std::pair<int, double>> sorted_entries;
  for (const auto& [row, val] : column_result) {
    if (val != 0.0) sorted_entries.emplace_back(row, val);
  }
  std::sort(sorted_entries.begin(), sorted_entries.end());

  int local_col = col_b - start_col;
  local_col_ptr[local_col + 1] = static_cast<int>(sorted_entries.size());
  std::cout << "  Column " << col_b << " has " << sorted_entries.size() << " non-zero entries" << std::endl;

  for (const auto& [row, val] : sorted_entries) {
    local_rows.push_back(row);
    local_values.push_back(val);
    std::cout << "  Adding entry: C[" << row << "][" << col_b << "] = " << val << std::endl;
  }
}

bool SparseMatmulTask::RunImpl() {
  int rank = world.rank();
  int size = world.size();
  std::cout << "\n[RunImpl] Rank " << rank << " started (total processes: " << size << ")" << std::endl;

  int base_cols = colsB / size;
  int extra_cols = colsB % size;
  int start_col =
      (rank < extra_cols) ? (base_cols + 1) * rank : (base_cols + 1) * extra_cols + base_cols * (rank - extra_cols);
  int end_col = start_col + ((rank < extra_cols) ? (base_cols + 1) : base_cols);
  int num_local_cols = end_col - start_col;

  std::cout << "[Process " << rank << "] Columns: [" << start_col << ", " << end_col << ")\n";
  std::cout << "[Process " << rank << "] Local columns to process: " << num_local_cols << std::endl;

  std::vector<double> local_values;
  std::vector<int> local_rows;
  std::vector<int> local_col_ptr(num_local_cols + 1, 0);

  int num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads <= 0) num_threads = 4;
  std::cout << "[RunImpl] Using " << num_threads << " threads" << std::endl;

  std::vector<std::vector<double>> thread_values(num_threads);
  std::vector<std::vector<int>> thread_rows(num_threads);
  std::vector<std::vector<int>> thread_col_ptrs(num_threads, std::vector<int>(num_local_cols + 1, 0));

  auto worker = [&](int thread_id) {
    std::cout << "[Thread " << thread_id << "] Started on rank " << rank << std::endl;
    for (int col = start_col + thread_id; col < end_col; col += num_threads) {
      std::cout << "[Thread " << thread_id << "] Processing column " << col << std::endl;
      ProcessColumn(col, start_col, thread_values[thread_id], thread_rows[thread_id], thread_col_ptrs[thread_id]);
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) t.join();

  std::cout << "\n[Aggregation] Rank " << rank << " merging thread data..." << std::endl;
  for (int local_col = 0; local_col < num_local_cols; ++local_col) {
    for (int t = 0; t < num_threads; ++t) {
      local_col_ptr[local_col + 1] += thread_col_ptrs[t][local_col + 1];
    }
  }
  for (int col = 1; col <= num_local_cols; ++col) {
    local_col_ptr[col] += local_col_ptr[col - 1];
  }
  for (int local_col = 0; local_col < num_local_cols; ++local_col) {
    for (int t = 0; t < num_threads; ++t) {
      int start = thread_col_ptrs[t][local_col];
      int end = thread_col_ptrs[t][local_col + 1];
      for (int i = start; i < end; ++i) {
        local_values.push_back(thread_values[t][i]);
        local_rows.push_back(thread_rows[t][i]);
      }
    }
    local_col_ptr[local_col + 1] = local_values.size();
  }

  std::cout << "[Aggregation] Rank " << rank << " results:\n";
  std::cout << "  Values: ";
  for (auto v : local_values) std::cout << v << " ";
  std::cout << "\n";
  std::cout << "  Rows: ";
  for (auto r : local_rows) std::cout << r << " ";
  std::cout << "\n";
  std::cout << "  Col_ptr: ";
  for (auto c : local_col_ptr) std::cout << c << " ";
  std::cout << "\n";

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
    std::cout << "\n[Final Aggregation] Root process building result..." << std::endl;
    C_values.clear();
    C_row_indices.clear();
    C_col_ptr.resize(colsB + 1, 0);

    std::cout << "Global column order: 0 to " << colsB - 1 << std::endl;
    for (int global_col = 0; global_col < colsB; ++global_col) {
      std::cout << "\nProcessing global column " << global_col << std::endl;
      for (int proc = 0; proc < size; ++proc) {
        if (global_col >= proc_start_cols[proc] && global_col < proc_end_cols[proc]) {
          int local_col = global_col - proc_start_cols[proc];
          int start = all_col_ptrs[proc][local_col];
          int end = all_col_ptrs[proc][local_col + 1];

          std::cout << "  Found in process " << proc << " (local_col=" << local_col << ", elements: " << (end - start)
                    << ")" << std::endl;

          for (int i = start; i < end; ++i) {
            C_values.push_back(all_values[proc][i]);
            C_row_indices.push_back(all_rows[proc][i]);
            std::cout << "    Added: C[" << all_rows[proc][i] << "][" << global_col << "] = " << all_values[proc][i]
                      << std::endl;
          }
          C_col_ptr[global_col + 1] = C_col_ptr[global_col] + (end - start);
          break;
        }
      }
    }

    std::cout << "\nFinal Result:" << std::endl;
    std::cout << "C_values: ";
    for (auto v : C_values) std::cout << v << " ";
    std::cout << "\n";
    std::cout << "C_row_indices: ";
    for (auto r : C_row_indices) std::cout << r << " ";
    std::cout << "\n";
    std::cout << "C_col_ptr: ";
    for (auto c : C_col_ptr) std::cout << c << " ";
    std::cout << "\n";
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() {
  std::cout << "[PostProcessing] Rank " << world.rank() << " completed" << std::endl;
  return true;
}

}  // namespace konkov_i_sparse_matmul_ccs_all