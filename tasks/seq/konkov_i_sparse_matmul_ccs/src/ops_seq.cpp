#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>

#include "core/task/include/task.hpp"
#include <unordered_map>

namespace konkov_i_sparse_matmul_ccs {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
    std::cerr << "Error: Matrices dimensions mismatch\n";
    return false;
  }
  if (A_col_ptr.empty() || B_col_ptr.empty()) {
    std::cerr << "Error: Empty matrix data\n";
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
  std::vector<std::unordered_map<int, double>> column_map(colsB);

  for (int col_B = 0; col_B < colsB; ++col_B) {
    for (int j = B_col_ptr[col_B]; j < B_col_ptr[col_B + 1]; ++j) {
      int row_B = B_row_indices[j];
      double val_B = B_values[j];

      if (row_B >= colsA || row_B + 1 > A_col_ptr.size()) continue;

      for (int k = A_col_ptr[row_B]; k < A_col_ptr[row_B + 1]; ++k) {
        if (k >= A_row_indices.size()) continue;

        int row_A = A_row_indices[k];
        double val_A = A_values[k];
        column_map[col_B][row_A] += val_A * val_B;
      }
    }
  }

  C_col_ptr.resize(colsB + 1, 0);
  int count = 0;
  for (int col = 0; col < colsB; ++col) {
    std::vector<int> rows;
    for (const auto& pair : column_map[col]) {
      if (pair.second != 0) {
        rows.push_back(pair.first);
      }
    }
    std::sort(rows.begin(), rows.end());

    for (int row : rows) {
      C_values.push_back(column_map[col][row]);
      C_row_indices.push_back(row);
      count++;
    }
    C_col_ptr[col + 1] = count;
  }

  return true;
}

bool SparseMatmulTask::PostProcessingImpl() { return true; }

}  // namespace konkov_i_sparse_matmul_ccs