#include "tbb/sadikov_I_SparseMatMul_TBB/include/SparseMatrix.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_tbb {
SparseMatrix SparseMatrix::Transpose(const SparseMatrix& matrix) {
  MatrixComponents components;
  auto max_size = std::max(matrix.GetRowsCount(), matrix.GetColumnsCount());
  std::vector<std::vector<double>> intermediate_values(max_size);
  std::vector<std::vector<int>> intermediate_indexes(max_size);
  int column_number = 0;
  int column_counter = 0;
  for (size_t i = 0; i < matrix.GetValues().size(); ++i) {
    if (column_counter == matrix.GetElementsSum()[column_number]) {
      column_number++;
    }
    column_counter++;
    intermediate_values[matrix.GetRows()[i]].emplace_back(matrix.GetValues()[i]);
    intermediate_indexes[matrix.GetRows()[i]].emplace_back(column_number);
  }
  for (size_t i = 0; i < intermediate_values.size(); ++i) {
    for (size_t j = 0; j < intermediate_values[i].size(); ++j) {
      components.m_values_.emplace_back(intermediate_values[i][j]);
      components.m_rows_.emplace_back(intermediate_indexes[i][j]);
    }
    if (i > 0) {
      components.m_elementsSum_.emplace_back(intermediate_values[i].size() + components.m_elementsSum_[i - 1]);
    } else {
      components.m_elementsSum_.emplace_back(intermediate_values[i].size());
    }
  }
  return SparseMatrix(matrix.GetColumnsCount(), matrix.GetRowsCount(), components);
}
double SparseMatrix::CalculateSum(const SparseMatrix& fmatrix, const SparseMatrix& smatrix,
                                  const std::vector<int>& felements_sum, const std::vector<int>& selements_sum,
                                  int i_index, int j_index) {
  int fmatrix_elements_count = GetElementsCount(j_index, felements_sum);
  int smatrix_elements_count = GetElementsCount(i_index, selements_sum);
  int fmatrix_start_index = j_index != 0 ? felements_sum[j_index] - fmatrix_elements_count : 0;
  int smatrix_start_index = i_index != 0 ? selements_sum[i_index] - smatrix_elements_count : 0;
  double sum = 0.0;
  for (int i = 0; i < fmatrix_elements_count; i++) {
    for (int j = 0; j < smatrix_elements_count; j++) {
      if (fmatrix.GetRows()[fmatrix_start_index + i] == smatrix.GetRows()[smatrix_start_index + j]) {
        sum += fmatrix.GetValues()[i + fmatrix_start_index] * smatrix.GetValues()[j + smatrix_start_index];
      }
    }
  }
  return sum;
}

SparseMatrix SparseMatrix::operator*(SparseMatrix& smatrix) const {
  std::map<int, MatrixComponents> components;
  auto fmatrix = Transpose(*this);
  const auto& felements_sum = fmatrix.GetElementsSum();
  const auto& selements_sum = smatrix.GetElementsSum();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, (selements_sum.size())),
                    [&](const tbb::blocked_range<size_t>& range) {
                      MatrixComponents component;
                      component.m_elementsSum_.resize(smatrix.GetColumnsCount());
                      for (auto i = range.begin(); i < range.end(); ++i) {
                        for (int j = 0; j < static_cast<int>(felements_sum.size()); ++j) {
                          double sum = CalculateSum(fmatrix, smatrix, felements_sum, selements_sum, i, j);
                          if (sum > kMEpsilon) {
                            component.m_values_.push_back(sum);
                            component.m_rows_.push_back(j);
                            component.m_elementsSum_[i]++;
                          }
                        }
                      }
                      for (size_t i = 1; i < component.m_elementsSum_.size(); ++i) {
                        component.m_elementsSum_[i] = component.m_elementsSum_[i] + component.m_elementsSum_[i - 1];
                      }
                      components[tbb::this_task_arena::current_thread_index()] = std::move(component);
                    });
  MatrixComponents result;
  for (const auto& [thread_num, data] : components) {
    std::copy(data.m_values_.begin(), data.m_values_.end(), std::back_inserter(result.m_values_));
    std::copy(data.m_rows_.begin(), data.m_rows_.end(), std::back_inserter(result.m_rows_));
    std::copy(data.m_elementsSum_.begin(), data.m_elementsSum_.end(), std::back_inserter(result.m_elementsSum_));
  }
  return SparseMatrix(smatrix.GetColumnsCount(), smatrix.GetColumnsCount(), result);
}

SparseMatrix SparseMatrix::MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values) {
  MatrixComponents compontents;
  compontents.m_elementsSum_.resize(columns_count);
  for (int i = 0; i < columns_count; ++i) {
    for (int j = 0; j < rows_count; ++j) {
      if (values[i + (columns_count * j)] != 0) {
        compontents.m_values_.emplace_back(values[i + (columns_count * j)]);
        compontents.m_rows_.emplace_back(j);
        compontents.m_elementsSum_[i] += 1;
      }
    }
    if (i != columns_count - 1) {
      compontents.m_elementsSum_[i + 1] = compontents.m_elementsSum_[i];
    }
  }
  return SparseMatrix(rows_count, columns_count, compontents);
}

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix) {
  std::vector<double> simpl_matrix(matrix.GetRowsCount() * matrix.GetColumnsCount(), 0.0);
  int column_number = 0;
  int column_counter = 0;
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

int SparseMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  }
  return elements_sum[index] - elements_sum[index - 1];
}

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count) {
  std::vector<double> answer(fmatrix_rows_count * smatrix_columns_count);
  for (int i = 0; i < fmatrix_rows_count; i++) {
    for (int j = 0; j < smatrix_columns_count; j++) {
      for (int n = 0; n < smatrix_rows_count; n++) {
        answer[j + (i * smatrix_columns_count)] +=
            fmatrix[(i * fmatrix_columns_count) + n] * smatrix[(n * smatrix_columns_count) + j];
      }
    }
  }
  return answer;
}
}  // namespace sadikov_i_sparse_matrix_multiplication_task_tbb