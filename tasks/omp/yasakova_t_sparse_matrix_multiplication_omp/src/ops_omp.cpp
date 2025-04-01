#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {
SparseMatrixFormat TransposeMatrixCRS(const SparseMatrixFormat &crs) {
  const auto new_cols = crs.RowCount();

  SparseMatrixFormat result;
  result.columns = new_cols;
  result.row_pointers.resize(crs.ColumnCount() + 2);
  result.column_indices.resize(crs.column_indices.size(), 0);
  result.elements.resize(crs.elements.size(), 0);

  for (uint32_t i = 0; i < crs.elements.size(); ++i) {
    ++result.row_pointers[crs.column_indices[i] + 2];
  }
  for (uint32_t i = 2; i < result.row_pointers.size(); ++i) {
    result.row_pointers[i] += result.row_pointers[i - 1];
  }
  for (uint32_t i = 0; i < new_cols; ++i) {
    for (uint32_t j = crs.row_pointers[i]; j < crs.row_pointers[i + 1]; ++j) {
      const auto new_index = result.row_pointers[crs.column_indices[j] + 1]++;
      result.elements[new_index] = crs.elements[j];
      result.column_indices[new_index] = i;
    }
  }
  result.row_pointers.pop_back();

  return result;
}
}  // namespace

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::ValidationImpl() {
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::PreProcessingImpl() {
  left_matrix_ = *reinterpret_cast<SparseMatrixFormat *>(task_data->inputs[0]);
  rhs_ = TransposeMatrixCRS(*reinterpret_cast<SparseMatrixFormat *>(task_data->inputs[1]));
  result_matrix_ = {};
  result_matrix_.row_pointers.resize(left_matrix_.RowCount() + 1);
  result_matrix_.columns = rhs_.RowCount();
  return true;
}

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::RunImpl() {
  const auto num_rows = left_matrix_.RowCount();
  const auto num_cols = rhs_.RowCount();

  std::vector<std::vector<std::tuple<std::complex<double>, uint32_t>>> buf(num_rows);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(num_rows); ++i) {
    for (uint32_t j = 0; j < num_cols; ++j) {
      auto ii = left_matrix_.row_pointers[i];
      auto ij = rhs_.row_pointers[j];
      std::complex<double> summul = 0.0;
      while (ii < left_matrix_.row_pointers[i + 1] && ij < rhs_.row_pointers[j + 1]) {
        if (left_matrix_.column_indices[ii] < rhs_.column_indices[ij]) {
          ++ii;
        } else if (left_matrix_.column_indices[ii] > rhs_.column_indices[ij]) {
          ++ij;
        } else {
          summul += left_matrix_.elements[ii++] * rhs_.elements[ij++];
        }
      }
      if (summul != 0.0) {
        buf[i].emplace_back(summul, j);
      }
    }
  }

  for (uint32_t i = 0; i < num_rows; i++) {
    result_matrix_.row_pointers[i + 1] = result_matrix_.row_pointers[i];
    for (const auto &[summul, j] : buf[i]) {
      result_matrix_.elements.push_back(summul);
      result_matrix_.column_indices.push_back(j);
      ++result_matrix_.row_pointers[i + 1];
    }
  }

  return true;
}

bool yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier::PostProcessingImpl() {
  *reinterpret_cast<SparseMatrixFormat *>(task_data->outputs[0]) = result_matrix_;
  return true;
}