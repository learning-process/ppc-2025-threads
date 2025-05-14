#include "all/konkov_i_sparse_matmul_ccs_all/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <core/util/include/util.hpp>
#include <map>
#include <thread>
#include <unordered_map>

namespace konkov_i_sparse_matmul_ccs_all {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data)
    : ppc::core::Task(std::move(task_data)), world_(boost::mpi::communicator()) {}

bool SparseMatmulTask::ValidationImpl() {
  if (world_.rank() != 0) return true;
  return colsA == rowsB && rowsA > 0 && colsB > 0 && !A_col_ptr.empty() && !B_col_ptr.empty();
}

bool SparseMatmulTask::PreProcessingImpl() {
  // Broadcast matrix dimensions
  int a_dims[2]{rowsA, colsA};
  int b_dims[2]{rowsB, colsB};
  boost::mpi::broadcast(world_, a_dims, 2, 0);
  boost::mpi::broadcast(world_, b_dims, 2, 0);
  rowsA = a_dims[0];
  colsA = a_dims[1];
  rowsB = b_dims[0];
  colsB = b_dims[1];

  // Broadcast matrix A
  std::vector<int> a_sizes(3);
  if (world_.rank() == 0) {
    a_sizes = {static_cast<int>(A_values.size()), static_cast<int>(A_row_indices.size()),
               static_cast<int>(A_col_ptr.size())};
  }
  boost::mpi::broadcast(world_, a_sizes, 0);

  if (world_.rank() != 0) {
    A_values.resize(a_sizes[0]);
    A_row_indices.resize(a_sizes[1]);
    A_col_ptr.resize(a_sizes[2]);
  }
  boost::mpi::broadcast(world_, A_values, 0);
  boost::mpi::broadcast(world_, A_row_indices, 0);
  boost::mpi::broadcast(world_, A_col_ptr, 0);

  // Broadcast matrix B
  std::vector<int> b_sizes(3);
  if (world_.rank() == 0) {
    b_sizes = {static_cast<int>(B_values.size()), static_cast<int>(B_row_indices.size()),
               static_cast<int>(B_col_ptr.size())};
  }
  boost::mpi::broadcast(world_, b_sizes, 0);

  if (world_.rank() != 0) {
    B_values.resize(b_sizes[0]);
    B_row_indices.resize(b_sizes[1]);
    B_col_ptr.resize(b_sizes[2]);
  }
  boost::mpi::broadcast(world_, B_values, 0);
  boost::mpi::broadcast(world_, B_row_indices, 0);
  boost::mpi::broadcast(world_, B_col_ptr, 0);

  C_col_ptr.assign(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();
  return true;
}

bool SparseMatmulTask::RunImpl() {
  std::vector<int> my_columns;
  for (int col_b = world_.rank(); col_b < colsB; col_b += world_.size()) {
    my_columns.push_back(col_b);
  }

  auto num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) num_threads = 4;

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
      for (auto&& [row, val] : column_result) {
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
  for (int t = 0; t < num_threads; ++t) threads.emplace_back(worker, t);
  for (auto& t : threads) t.join();

  std::vector<std::tuple<int, std::vector<double>, std::vector<int>>> local_results;
  for (auto& res : thread_results) {
    local_results.insert(local_results.end(), res.begin(), res.end());
  }

  if (world_.rank() != 0) {
    world_.send(0, 0, local_results);
  } else {
    std::map<int, std::pair<std::vector<double>, std::vector<int>>> results_map;

    for (auto& [col, vals, rows] : local_results) {
      results_map[col] = {vals, rows};
    }

    for (int src = 1; src < world_.size(); ++src) {
      std::vector<std::tuple<int, std::vector<double>, std::vector<int>>> remote_results;
      world_.recv(src, 0, remote_results);

      for (auto& [col, vals, rows] : remote_results) {
        results_map[col] = {vals, rows};
      }
    }

    C_col_ptr.resize(colsB + 1, 0);
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