#include "omp/Sadikov_I_SparesMatrixMultiplication_OMP/include/SparesMatrix.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_omp {
SparesMatrix SparesMatrix::Transpose(const SparesMatrix& matrix) {
  std::vector<double> val;
  std::vector<int> rows;
  std::vector<int> elem_sum;
  auto max_size = std::max(matrix.GetRowsCount(), matrix.GetColumnsCount());
  std::vector<std::vector<double>> intermediate_values(max_size);
  std::vector<std::vector<int>> intermediate_indexes(max_size);
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
double SparesMatrix::CalculateSum(SparesMatrix& fmatrix, SparesMatrix& smatrix, const std::vector<int>& felements_sum,
                                  const std::vector<int>& selements_sum, int i_index, int j_index) {
  auto fmatrix_elements_count = GetElementsCount(j_index, felements_sum);
  auto smatrix_elements_count = GetElementsCount(i_index, selements_sum);
  auto fmatrix_start_index = j_index != 0 ? felements_sum[j_index] - fmatrix_elements_count : 0;
  auto smatrix_start_index = i_index != 0 ? selements_sum[i_index] - smatrix_elements_count : 0;
  double sum = 0.0;
  for (auto n = 0; n < fmatrix_elements_count; n++) {
    for (auto n2 = 0; n2 < smatrix_elements_count; n2++) {
      if (fmatrix.GetRows()[fmatrix_start_index + n] == smatrix.GetRows()[smatrix_start_index + n2]) {
        sum += fmatrix.GetValues()[n + fmatrix_start_index] * smatrix.GetValues()[n2 + smatrix_start_index];
      }
    }
  }
  return sum;
}

SparesMatrix SparesMatrix::operator*(SparesMatrix& smatrix) {
  std::vector<double> values;
  std::vector<int> rows;
  std::vector<int> elements_sum(smatrix.GetColumnsCount());
  auto fmatrix = Transpose(*this);
  const auto& felements_sum = fmatrix.GetElementsSum();
  const auto& selements_sum = smatrix.GetElementsSum();
  std::vector<std::vector<std::pair<double, int>>> intermediate_values(18);
#pragma omp parallel
  {
    std::vector<std::pair<double, int>> thread_data;
#pragma omp for
    for (auto i = 0; i < static_cast<int>(selements_sum.size()); ++i) {
      for (auto j = 0; j < static_cast<int>(felements_sum.size()); ++j) {
        double sum = CalculateSum(fmatrix, smatrix, felements_sum, selements_sum, i, j);
        if (sum > kMEpsilon) {
          thread_data.emplace_back(sum, j);
          elements_sum[i]++;
        }
      }
    }
    intermediate_values[omp_get_thread_num()] = thread_data;
  }
  for (auto&& it : intermediate_values) {
    for (auto&& it2 : it) {
      values.emplace_back(it2.first);
      rows.emplace_back(it2.second);
    }
  }
  for (size_t i = 1; i < elements_sum.size(); ++i) {
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
  for (size_t i = 0; i < matrix.GetValues().size(); ++i) {
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

int SparesMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  }
  return elements_sum[index] - elements_sum[index - 1];
}

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count) {
  std::vector<double> answer(fmatrix_rows_count * smatrix_columns_count);
#pragma omp parallel
  {
#pragma omp for
    for (auto i = 0; i < fmatrix_rows_count; i++) {
      for (auto j = 0; j < smatrix_columns_count; j++) {
        for (auto n = 0; n < smatrix_rows_count; n++) {
          answer[j + (i * smatrix_columns_count)] +=
              fmatrix[(i * fmatrix_columns_count) + n] * smatrix[(n * smatrix_columns_count) + j];
        }
      }
    }
  }
  return answer;
}
}  // namespace sadikov_i_sparse_matrix_multiplication_task_omp