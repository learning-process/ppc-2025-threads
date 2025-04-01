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

  std::complex<double>& AccessElement(uint32_t row, uint32_t col) noexcept { return elements[(row * num_cols) + col]; }

  [[nodiscard]] std::complex<double>& AccessElement(uint32_t row, uint32_t col) const noexcept {
    return elements[(row * num_cols) + col];
  }

  bool operator==(const MatrixStructure& other) const noexcept {
    return num_rows == other.num_rows && num_cols == other.num_cols && elements == other.elements;
  }
};

inline MatrixStructure MatrixMultiply(const MatrixStructure& mat_a, const MatrixStructure& mat_b) {
  MatrixStructure result{.num_rows = mat_a.num_rows,
                         .num_cols = mat_b.num_cols,
                         .elements = std::vector<std::complex<double>>(mat_a.num_rows * mat_b.num_cols, 0.0)};
  for (uint32_t i = 0; i < mat_a.num_rows; ++i) {
    for (uint32_t k = 0; k < mat_b.num_rows; ++k) {
      const auto temp = mat_a.AccessElement(i, k);
      if (temp == 0.0){
        continue;
      }
      for (uint32_t j = 0; j < mat_b.num_cols; ++j) {
        result.AccessElement(i, j) += temp * mat_b.AccessElement(k, j);
      }
    }
  }
  return result;
}

struct SparseMatrixFormat {
  std::vector<std::complex<double>> elements;
  uint32_t columns;
  std::vector<uint32_t> row_pointers;
  std::vector<uint32_t> column_indices;

  [[nodiscard]] uint32_t RowCount() const noexcept { return row_pointers.empty() ? 0 : row_pointers.size() - 1; }
  [[nodiscard]] uint32_t ColumnCount() const noexcept { return columns; }

  bool operator==(const SparseMatrixFormat& other) const noexcept {
    return columns == other.columns && row_pointers == other.row_pointers && column_indices == other.column_indices &&
           elements == other.elements;
  }
};

inline SparseMatrixFormat ConvertToCRS(const MatrixStructure& matrix) {
  SparseMatrixFormat result;
  result.row_pointers.reserve(matrix.num_rows + 1);
  result.row_pointers.push_back(0);
  result.columns = matrix.num_cols;

  for (uint32_t row = 0; row < matrix.num_rows; ++row) {
    uint32_t nz = 0;
    for (uint32_t col = 0; col < matrix.num_cols; ++col) {
      const auto& element = matrix.AccessElement(row, col);
      if (element != 0.0) {
        ++nz;
        result.column_indices.push_back(col);
        result.elements.push_back(element);
      }
    }
    result.row_pointers.push_back(result.row_pointers.back() + nz);
  }

  return result;
}

inline MatrixStructure ConvertFromCRS(const SparseMatrixFormat& crs) {
  MatrixStructure matrix{.num_rows = crs.RowCount(),
                         .num_cols = crs.ColumnCount(),
                         .elements = std::vector<std::complex<double>>(crs.RowCount() * crs.ColumnCount(), 0.0)};
  for (uint32_t row = 0; row < matrix.num_rows; ++row) {
    for (uint32_t i = crs.row_pointers[row]; i < crs.row_pointers[row + 1]; ++i) {
      matrix.AccessElement(row, crs.column_indices[i]) = crs.elements[i];
    }
  }
  return matrix;
}

namespace yasakova_t_sparse_matrix_multiplication_omp {

class SparseMatrixMultiplier : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiplier(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrixFormat left_matrix_;
  SparseMatrixFormat rhs_;
  SparseMatrixFormat result_matrix_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication_omp