#pragma once

#include <omp.h>

#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_tbb {
class SparseMatrix {
  constexpr static double kMEpsilon = 0.000001;
  struct MatrixComponents {
    std::vector<double> m_values_;
    std::vector<int> m_rows_;
    std::vector<int> m_elementsSum_;
    void Resize(size_t values_size, size_t sums_size) noexcept {
      m_values_.resize(values_size);
      m_rows_.resize(values_size);
      m_elementsSum_.resize(sums_size);
    }
  };
  int m_rowsCount_ = 0;
  int m_columnsCount_ = 0;
  MatrixComponents m_compontents_;

  static SparseMatrix Transpose(const SparseMatrix& matrix);
  static int GetElementsCount(int index, const std::vector<int>& elements_sum);
  static double CalculateSum(const SparseMatrix& fmatrix, const SparseMatrix& smatrix,
                             const std::vector<int>& felements_sum, const std::vector<int>& selements_sum, int i_index,
                             int j_index);

 public:
  SparseMatrix() = default;
  explicit SparseMatrix(int rows_count, int columns_count, MatrixComponents components) noexcept
      : m_rowsCount_(rows_count), m_columnsCount_(columns_count), m_compontents_(components) {};
  [[nodiscard]] const std::vector<double>& GetValues() const noexcept { return m_compontents_.m_values_; }
  [[nodiscard]] const std::vector<int>& GetRows() const noexcept { return m_compontents_.m_rows_; }
  [[nodiscard]] const std::vector<int>& GetElementsSum() const noexcept { return m_compontents_.m_elementsSum_; }
  [[nodiscard]] int GetColumnsCount() const noexcept { return m_columnsCount_; }
  [[nodiscard]] int GetRowsCount() const noexcept { return m_rowsCount_; }
  static SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values);
  SparseMatrix operator*(SparseMatrix& smatrix) const noexcept(false);
};

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix);

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count);
}  // namespace sadikov_i_sparse_matrix_multiplication_task_tbb