#include "stl/konkov_i_sparse_matmul_ccs_stl/include/ops_stl.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_stl {

SparseMatmulTaskSTL::SparseMatmulTaskSTL(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTaskSTL::ValidationImpl() {
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
    return false;
  }
  if (A_col_ptr.empty() || B_col_ptr.empty()) {
    return false;
  }
  return true;
}

bool SparseMatmulTaskSTL::PreProcessingImpl() {
  C_col_ptr.resize(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();
  return true;
}

bool SparseMatmulTaskSTL::RunImpl() {
  std::vector<std::map<int, double>> column_map(colsB);

  for (int col_b = 0; col_b < colsB; ++col_b) {
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
        column_map[col_b][row_a] += val_a * val_b;
      }
    }
  }

  C_col_ptr.resize(colsB + 1, 0);
  int count = 0;
  for (int col = 0; col < colsB; ++col) {
    for (const auto& [row, val] : column_map[col]) {
      if (val != 0) {
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