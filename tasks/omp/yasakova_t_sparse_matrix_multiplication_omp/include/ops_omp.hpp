#pragma once

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct MatrixStructure {
  uint32_t num_rows;
  uint32_t num_cols;
  std::vector<std::complex<double>> elements;

  std::complex<double>& AccessElement(uint32_t row, uint32_t col) { return elements[(row * num_cols) + col]; }

  bool operator==(const MatrixStructure& other) const noexcept {
    return num_rows == other.num_rows && num_cols == other.num_cols && elements == other.elements;
  }
};

inline MatrixStructure MultiplyMatrices(MatrixStructure& left_matrix, MatrixStructure& right_matrix) {
  MatrixStructure result{.num_rows = left_matrix.num_rows, .num_cols = right_matrix.num_cols, .elements = std::vector<std::complex<double>>(left_matrix.num_rows * right_matrix.num_cols)};
  for (uint32_t row_idx = 0; row_idx < left_matrix.num_rows; row_idx++) {
    for (uint32_t col_idx = 0; col_idx < right_matrix.num_cols; col_idx++) {
      result.AccessElement(row_idx, col_idx) = 0;
      for (uint32_t k_idx = 0; k_idx < right_matrix.num_rows; k_idx++) {
        result.AccessElement(row_idx, col_idx) += left_matrix.AccessElement(row_idx, k_idx) * right_matrix.AccessElement(k_idx, col_idx);
      }
    }
  }
  return result;
}

struct SparseMatrixCRS {
  std::vector<std::complex<double>> elements;

  uint32_t total_columns;
  std::vector<uint32_t> row_pointers;
  std::vector<uint32_t> column_indices;

  //

  [[nodiscard]] uint32_t GetRowCount() const { return row_pointers.size() - 1; }
  [[nodiscard]] uint32_t GetColumnCount() const { return total_columns; }

  bool operator==(const SparseMatrixCRS& other) const noexcept {
    return total_columns == other.total_columns && row_pointers == other.row_pointers && column_indices == other.column_indices && elements == other.elements;
  }
};

inline SparseMatrixCRS ConvertToCRS(const MatrixStructure& matrix) {
  SparseMatrixCRS result;
  result.row_pointers.resize(matrix.num_rows + 1);
  result.total_columns = matrix.num_cols;

  uint32_t row_idx = 0;
  for (uint32_t row = 0; row < matrix.num_rows; ++row) {
    uint32_t non_zero_count = 0;
    for (uint32_t col = 0; col < matrix.num_cols; ++col) {
      if (const auto& element = matrix.elements[row_idx++]; element != 0.0) {
        ++non_zero_count;
        result.column_indices.push_back(col);
        result.elements.push_back(element);
      }
    }
    result.row_pointers[row + 1] = result.row_pointers[row] + non_zero_count;
  }

  return result;
}

inline MatrixStructure ConvertFromCRS(const SparseMatrixCRS& crs) {
  MatrixStructure matrix{.num_rows = crs.GetRowCount(),
                .num_cols = crs.GetColumnCount(),
                .elements = std::vector<std::complex<double>>(crs.GetRowCount() * crs.GetColumnCount())};
  for (uint32_t row = 0; row < matrix.num_rows; ++row) {
    for (uint32_t row_idx = crs.row_pointers[row]; row_idx < crs.row_pointers[row + 1]; ++row_idx) {
      matrix.AccessElement(row, crs.column_indices[row_idx]) = crs.elements[row_idx];
    }
  }
  return matrix;
}

namespace yasakova_t_sparse_matrix_multiplication_omp {

class MatrixMultiplicationTask : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool Validate() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrixCRS left_matrix_;
  SparseMatrixCRS rhs_;
  SparseMatrixCRS res_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication_omp