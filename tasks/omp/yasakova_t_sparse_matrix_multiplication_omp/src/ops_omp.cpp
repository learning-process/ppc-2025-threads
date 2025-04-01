#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {
SparseMatrixCRS TransposeMatrixCRS(const SparseMatrixCRS &crs) {
  const auto new_cols = crs.GetRowCount();

  SparseMatrixCRS result;
  result.total_columns = new_cols;
  result.row_pointers.resize(crs.GetColumnCount() + 2);
  result.column_indices.resize(crs.column_indices.size(), 0);
  result.elements.resize(crs.elements.size(), 0);

  for (uint32_t row_idx = 0; row_idx < crs.elements.size(); ++row_idx) {
    ++result.row_pointers[crs.column_indices[row_idx] + 2];
  }
  for (uint32_t row_idx = 2; row_idx < result.row_pointers.size(); ++row_idx) {
    result.row_pointers[row_idx] += result.row_pointers[row_idx - 1];
  }
  for (uint32_t row_idx = 0; row_idx < new_cols; ++row_idx) {
    for (uint32_t col_idx = crs.row_pointers[row_idx]; col_idx < crs.row_pointers[row_idx + 1]; ++col_idx) {
      const auto new_index = result.row_pointers[crs.column_indices[col_idx] + 1]++;
      result.elements[new_index] = crs.elements[col_idx];
      result.column_indices[new_index] = row_idx;
    }
  }
  result.row_pointers.pop_back();

  return result;
}
}  // namespace

bool yasakova_t_sparse_matrix_multiplication_omp::MatrixMultiplicationTask::Validate() {
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool yasakova_t_sparse_matrix_multiplication_omp::MatrixMultiplicationTask::PreProcessingImpl() {
  left_matrix_ = *reinterpret_cast<SparseMatrixCRS *>(task_data->inputs[0]);
  rhs_ = TransposeMatrixCRS(*reinterpret_cast<SparseMatrixCRS *>(task_data->inputs[1]));
  res_ = {};
  res_.row_pointers.resize(left_matrix_.GetRowCount() + 1);
  res_.total_columns = rhs_.GetRowCount();
  return true;
}

bool yasakova_t_sparse_matrix_multiplication_omp::MatrixMultiplicationTask::RunImpl() {
  const auto num_rows = left_matrix_.GetRowCount();
  const auto num_cols = rhs_.GetRowCount();

  std::vector<std::vector<std::tuple<std::complex<double>, uint32_t>>> buf(num_rows);

#pragma omp parallel for
  for (int row_idx = 0; row_idx < static_cast<int>(num_rows); ++row_idx) {
    for (uint32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
      auto ii = left_matrix_.row_pointers[row_idx];
      auto ij = rhs_.row_pointers[col_idx];
      std::complex<double> summul = 0.0;
      while (ii < left_matrix_.row_pointers[row_idx + 1] && ij < rhs_.row_pointers[col_idx + 1]) {
        if (left_matrix_.column_indices[ii] < rhs_.column_indices[ij]) {
          ++ii;
        } else if (left_matrix_.column_indices[ii] > rhs_.column_indices[ij]) {
          ++ij;
        } else {
          summul += left_matrix_.elements[ii++] * rhs_.elements[ij++];
        }
      }
      if (summul != 0.0) {
        buf[row_idx].emplace_back(summul, col_idx);
      }
    }
  }

  for (uint32_t row_idx = 0; row_idx < num_rows; row_idx++) {
    res_.row_pointers[row_idx + 1] = res_.row_pointers[row_idx];
    for (const auto &[summul, col_idx] : buf[row_idx]) {
      res_.elements.push_back(summul);
      res_.column_indices.push_back(col_idx);
      ++res_.row_pointers[row_idx + 1];
    }
  }

  return true;
}

bool yasakova_t_sparse_matrix_multiplication_omp::MatrixMultiplicationTask::PostProcessingImpl() {
  *reinterpret_cast<SparseMatrixCRS *>(task_data->outputs[0]) = res_;
  return true;
}