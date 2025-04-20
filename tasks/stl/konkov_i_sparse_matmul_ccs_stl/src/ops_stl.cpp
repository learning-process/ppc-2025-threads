#include "stl/konkov_i_sparse_matmul_ccs_stl/include/ops_stl.hpp"

#include <cstddef>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_stl {

SparseMatmulTaskSTL::SparseMatmulTaskSTL(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTaskSTL::ValidationImpl() {
  return !(colsA != rowsB || rowsA <= 0 || colsB <= 0 || A_col_ptr.empty() || B_col_ptr.empty());
}

bool SparseMatmulTaskSTL::PreProcessingImpl() {
  C_col_ptr.resize(colsB + 1, 0);
  C_values.clear();
  C_row_indices.clear();
  return true;
}

struct ThreadResult {
  std::map<int, double> column_map;
  int col_start, col_end;
};

void compute_thread(const std::vector<double>& A_values, const std::vector<int>& A_row_indices,
                    const std::vector<int>& A_col_ptr, const std::vector<double>& B_values,
                    const std::vector<int>& B_row_indices, const std::vector<int>& B_col_ptr, int colsA,
                    ThreadResult& result) {
  for (int col_b = result.col_start; col_b < result.col_end; ++col_b) {
    for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
      int row_b = B_row_indices[j];
      double val_b = B_values[j];
      if (row_b >= colsA) continue;

      for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
        if (static_cast<size_t>(k) >= A_row_indices.size()) continue;
        int row_a = A_row_indices[k];
        double val_a = A_values[k];
        result.column_map[row_a] += val_a * val_b;
      }
    }
  }
}

bool SparseMatmulTaskSTL::RunImpl() {
  const int num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  std::vector<ThreadResult> thread_results(num_threads);

  const int cols_per_thread = (colsB + num_threads - 1) / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    thread_results[i].col_start = i * cols_per_thread;
    thread_results[i].col_end = std::min((i + 1) * cols_per_thread, colsB);
    threads.emplace_back(compute_thread, std::ref(A_values), std::ref(A_row_indices), std::ref(A_col_ptr),
                         std::ref(B_values), std::ref(B_row_indices), std::ref(B_col_ptr), colsA,
                         std::ref(thread_results[i]));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  std::mutex mtx;
  int count = 0;
  for (int col = 0; col < colsB; ++col) {
    for (const auto& [row, val] : thread_results[col / cols_per_thread].column_map) {
      if (val != 0) {
        std::lock_guard<std::mutex> lock(mtx);
        C_values.push_back(val);
        C_row_indices.push_back(row);
        count++;
      }
    }
    C_col_ptr[col + 1] = count;
  }

  return true;
}

bool SparseMatmulTaskSTL::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs_stl