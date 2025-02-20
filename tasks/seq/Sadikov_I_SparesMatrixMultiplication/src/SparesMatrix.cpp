#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

#include <iostream>
#include <vector>

SparesMatrix SparesMatrix::Transpose(const SparesMatrix& matrix) {
  std::vector<double> val;
  std::vector<int> rows;
  std::vector<int> elem_sum;
  std::vector<std::vector<double>> intermediate_values(matrix.GetRowsCount());
  std::vector<std::vector<int>> intermediate_indexes(matrix.GetColumnsCount());
  auto column_number = 0;
  auto column_counter = 0;
  for (auto i = 0; i < static_cast<int>(matrix.GetValues().size()); ++i) {
    if (column_counter == matrix.GetElementsSum()[column_number]) {
      column_number++;
    }
    column_counter++;
    intermediate_values[matrix.GetRows()[i]].emplace_back(matrix.GetValues()[i]);
    intermediate_indexes[matrix.GetRows()[i]].emplace_back(column_number);
  }
  for (auto i = 0; i < static_cast<int>(intermediate_values.size()); ++i) {
    for (auto j = 0; j < static_cast<int>(intermediate_values[i].size()); ++j) {
      val.emplace_back(intermediate_values[i][j]);
      rows.emplace_back(intermediate_indexes[i][j]);
    }
    if (i > 0) {
      elem_sum.emplace_back(intermediate_values[i].size() + elem_sum[i - 1]);
    } else {
      elem_sum.emplace_back(intermediate_values[i].size());
    }
  }
  return SparesMatrix(matrix.GetColumnsCount(), matrix.GetRowsCount(), val, rows, elem_sum);
}

SparesMatrix SparesMatrix::operator*(const SparesMatrix& smatrix) const {
  std::vector<double> values;
  std::vector<int> rows;
  std::vector<int> elements_sum(smatrix.GetColumnsCount());
  auto fmatrix = Transpose(*this);
  const auto& felements_sum = fmatrix.GetElementsSum();
  const auto& selements_sum = smatrix.GetElementsSum();

  for (auto i = 0; i < static_cast<int>(selements_sum.size()); ++i) {
    for (auto j = 0; j < static_cast<int>(felements_sum.size()); ++j) {
      auto sum = 0.0;
      auto fmatrix_elements_count = GetElementsCount(j, felements_sum);
      auto smatrix_elements_count = GetElementsCount(i, selements_sum);
      auto fmatrix_start_index = j != 0 ? felements_sum[j] - fmatrix_elements_count : 0;
      auto smatrix_start_index = i != 0 ? selements_sum[i] - smatrix_elements_count : 0;

      for (auto n = 0; n < fmatrix_elements_count; n++) {
        for (auto n2 = 0; n2 < smatrix_elements_count; n2++) {
          if (fmatrix.GetRows()[fmatrix_start_index + n] == smatrix.GetRows()[smatrix_start_index + n2]) {
            sum += fmatrix.GetValues()[n + fmatrix_start_index] * smatrix.GetValues()[n2 + smatrix_start_index];
          }
        }
      }
      if (sum > kMEpsilon) {
        values.emplace_back(sum);
        rows.emplace_back(j);
        elements_sum[i]++;
      }
    }
  }
  for (auto i = 1; i < static_cast<int>(elements_sum.size()); ++i) {
    elements_sum[i] = elements_sum[i] + elements_sum[i - 1];
  }
  return SparesMatrix(smatrix.GetColumnsCount(), smatrix.GetColumnsCount(), values, rows, elements_sum);
}

SparesMatrix MatrixToSpares(int rows_count, int columns_count, const std::vector<double>& values) {
  std::vector<double> val;
  std::vector<int> sums(columns_count, 0);
  std::vector<int> rows;
  for (auto i = 0; i < columns_count; ++i) {
    for (auto j = 0; j < rows_count; ++j) {
      if (values[i + (columns_count * j)] != 0) {
        val.emplace_back(values[i + (columns_count * j)]);
        rows.emplace_back(j);
        sums[i] += 1;
      }
    }
    if (i != columns_count - 1) {
      sums[i + 1] = sums[i];
    }
  }
  return SparesMatrix(rows_count, columns_count, val, rows, sums);
}

std::vector<double> FromSparesMatrix(const SparesMatrix& matrix) {
  std::vector<double> simpl_matrix(matrix.GetRowsCount() * matrix.GetColumnsCount(), 0.0);
  auto column_number = 0;
  auto column_counter = 0;
  for (auto i = 0; i < static_cast<int>(matrix.GetValues().size()); ++i) {
    if (column_counter >= matrix.GetElementsSum()[column_number]) {
      column_number++;
    }
    column_counter++;
    if (column_number > 0 && matrix.GetElementsSum()[column_number] - matrix.GetElementsSum()[column_number - 1] == 0) {
      column_number++;
    }
    simpl_matrix[column_number + (matrix.GetRows()[i] * matrix.GetColumnsCount())] = matrix.GetValues()[i];
  }
  return simpl_matrix;
}

std::ostream& operator<<(std::ostream& os, const SparesMatrix& matrix) {
  os << "VALUES" << '\n';
  for (auto i = 0; i < static_cast<int>(matrix.GetValues().size()); ++i) {
    os << matrix.GetValues()[i] << " ";
  }
  os << '\n' << "ROWS" << '\n';
  for (auto i = 0; i < static_cast<int>(matrix.GetRows().size()); ++i) {
    os << matrix.GetRows()[i] << " ";
  }
  os << '\n' << "ElementsSum" << '\n';
  for (auto i = 0; i < static_cast<int>(matrix.GetElementsSum().size()); ++i) {
    os << matrix.GetElementsSum()[i] << " ";
  }
  os << '\n';
  return os;
}

int SparesMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  } else {
    return elements_sum[index] - elements_sum[index - 1];
  }
}