#pragma once

#include <iostream>
#include <vector>

class SparesMatrix {
  constexpr static double m_epsilon = 0.000001;
  int m_rowsCount = 0;
  int m_columnsCount = 0;
  std::vector<double> m_values;
  std::vector<int> m_rows;
  std::vector<int> m_elementsSum;
  SparesMatrix Transpose(const SparesMatrix& matrix);

 public:
  SparesMatrix() = default;
  explicit SparesMatrix(int rowsCount, int ColumnsCount, const std::vector<double>& values,
                        const std::vector<int>& rows, const std::vector<int>& elementSum) noexcept
      : m_rowsCount(rowsCount),
        m_columnsCount(ColumnsCount),
        m_values(values),
        m_rows(rows),
        m_elementsSum(elementSum){};
  const std::vector<double>& GetValues() const noexcept { return m_values; }
  const std::vector<int>& GetRows() const noexcept { return m_rows; }
  const std::vector<int>& GetElementsSum() const noexcept { return m_elementsSum; }
  int GetColumnsCount() const noexcept { return m_columnsCount; }
  int GetRowsCount() const noexcept { return m_rowsCount; }
  SparesMatrix operator*(const SparesMatrix& smatrix) noexcept(false);
};

SparesMatrix MatrixToSpares(int rowsCount, int columnsCount, const std::vector<double>& values);

std::vector<double> FromSparesMatrix(const SparesMatrix& matrix);

std::ostream& operator<<(std::ostream& os, const SparesMatrix& matrix);