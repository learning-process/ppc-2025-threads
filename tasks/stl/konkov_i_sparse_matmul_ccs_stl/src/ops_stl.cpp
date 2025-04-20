#include "stl/konkov_i_sparse_matmul_ccs_stl/include/ops_stl.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace konkov_i_sparse_matmul_ccs_stl {

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
  C_col_ptr.resize(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();
  return true;
}

bool SparseMatmulTask::RunImpl() {
  const int num_threads = std::thread::hardware_concurrency();
  std::vector<std::vector<double>> thread_C_values(num_threads);
  std::vector<std::vector<int>> thread_C_row_indices(num_threads);
  std::vector<std::vector<int>> thread_C_col_ptr(num_threads, std::vector<int>(colsB + 1, 0));
  std::mutex result_mutex;

  auto worker = [&](int thread_id, int col_start, int col_end) {
    std::vector<std::unordered_map<int, double>> column_map(colsB);

    for (int col_b = col_start; col_b < col_end; ++col_b) {
      for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
        int row_b = B_row_indices[j];
        double val_b = B_values[j];

        if (row_b >= colsA) continue;

        for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
          if (static_cast<size_t>(k) >= A_row_indices.size()) continue;

          int row_a = A_row_indices[k];
          double val_a = A_values[k];
          column_map[col_b][row_a] += val_a * val_b;
        }
      }
    }

    int count = 0;
    for (int col = col_start; col < col_end; ++col) {
      std::vector<int> rows;
      for (const auto& pair : column_map[col]) {
        if (pair.second != 0.0) {
          rows.push_back(pair.first);
        }
      }
      std::ranges::sort(rows);

      for (int row : rows) {
        thread_C_values[thread_id].push_back(column_map[col][row]);
        thread_C_row_indices[thread_id].push_back(row);
        count++;
      }
      thread_C_col_ptr[thread_id][col + 1] = count;
    }
  };

  std::vector<std::thread> threads;
  int cols_per_thread = colsB / num_threads;
  int remainder = colsB % num_threads;
  int current = 0;

  for (int t = 0; t < num_threads; ++t) {
    int start = current;
    int end = start + cols_per_thread + (t < remainder ? 1 : 0);
    current = end;
    threads.emplace_back(worker, t, start, end);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  C_col_ptr.assign(colsB + 1, 0);
  for (int t = 0; t < num_threads; ++t) {
    C_values.insert(C_values.end(), thread_C_values[t].begin(), thread_C_values[t].end());
    C_row_indices.insert(C_row_indices.end(), thread_C_row_indices[t].begin(), thread_C_row_indices[t].end());

    for (int i = 1; i <= colsB; ++i) {
      C_col_ptr[i] += thread_C_col_ptr[t][i];
    }
  }

  for (int i = 1; i <= colsB; ++i) {
    C_col_ptr[i] += C_col_ptr[i - 1];
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_stl
