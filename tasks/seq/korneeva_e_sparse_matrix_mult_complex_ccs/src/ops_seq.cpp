#include "seq/korneeva_e_sparse_matrix_mult_complex_ccs/include/ops_seq.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_sparse_matrix_mult_complex_ccs_seq {

bool SparseMatrixMultComplexCCS::PreProcessingImpl() {
  matrix1_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  matrix2_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);
  result_ = SparseMatrixCCS(matrix1_->rows, matrix2_->cols, 0);
  return true;
}

bool SparseMatrixMultComplexCCS::ValidationImpl() {
  if (task_data->inputs.size() != 2 || task_data->outputs.size() != 1) {
    return false;
  }
  if (matrix1_ == nullptr || matrix2_ == nullptr) {
    return false;
  }
  if (matrix1_->cols != matrix2_->rows) {
    return false;
  }
  if (matrix1_->rows <= 0 || matrix1_->cols <= 0 || matrix2_->rows <= 0 || matrix2_->cols <= 0) {
    return false;
  }
  return true;
}

bool SparseMatrixMultComplexCCS::RunImpl() {
  std::vector<Complex> temp_values;
  std::vector<int> temp_row_indices;
  std::vector<int> temp_col_offsets(1, 0);

  for (int j = 0; j < matrix2_->cols; j++) {
    ComputeColumn(j, temp_values, temp_row_indices, temp_col_offsets);
  }

  result_.values = std::move(temp_values);
  result_.row_indices = std::move(temp_row_indices);
  result_.col_offsets = std::move(temp_col_offsets);
  result_.nnz = static_cast<int>(result_.values.size());
  return true;
}

void SparseMatrixMultComplexCCS::ComputeColumn(int col_idx, std::vector<Complex>& values, std::vector<int>& row_indices,
                                               std::vector<int>& col_offsets) {
  int col_start2 = matrix2_->col_offsets[col_idx];
  int col_end2 = matrix2_->col_offsets[col_idx + 1];

  for (int i = 0; i < matrix1_->rows; i++) {
    Complex sum = ComputeElement(i, col_start2, col_end2);
    if (sum != Complex(0.0, 0.0)) {
      values.push_back(sum);
      row_indices.push_back(i);
    }
  }
  col_offsets.push_back(static_cast<int>(values.size()));
}

Complex SparseMatrixMultComplexCCS::ComputeElement(int row_idx, int col_start2, int col_end2) {
  Complex sum(0.0, 0.0);
  for (int k = 0; k < matrix1_->cols; k++) {
    int col_start1 = matrix1_->col_offsets[k];
    int col_end1 = matrix1_->col_offsets[k + 1];
    sum += ComputeContribution(row_idx, k, col_start1, col_end1, col_start2, col_end2);
  }
  return sum;
}

Complex SparseMatrixMultComplexCCS::ComputeContribution(int row_idx, int k, int col_start1, int col_end1,
                                                        int col_start2, int col_end2) {
  Complex contribution(0.0, 0.0);
  for (int p = col_start1; p < col_end1; p++) {
    if (matrix1_->row_indices[p] == row_idx) {
      for (int q = col_start2; q < col_end2; q++) {
        if (matrix2_->row_indices[q] == k) {
          contribution += matrix1_->values[p] * matrix2_->values[q];
        }
      }
    }
  }
  return contribution;
}

bool SparseMatrixMultComplexCCS::PostProcessingImpl() {
  *reinterpret_cast<SparseMatrixCCS*>(task_data->outputs[0]) = result_;  // Копирование
  return true;
}

SparseMatrixCCS CreateRandomMatrix(int rows, int cols, int max_nnz, std::mt19937& gen) {
  SparseMatrixCCS matrix(rows, cols, 0);
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  std::uniform_int_distribution<> row_dis(0, rows - 1);

  std::vector<std::vector<Complex>> temp(rows, std::vector<Complex>(cols, {0.0, 0.0}));
  int nnz = 0;
  while (nnz < max_nnz && nnz < rows * cols) {
    int r = row_dis(gen);
    int c = row_dis(gen) % cols;
    if (temp[r][c] == Complex(0.0, 0.0)) {
      temp[r][c] = Complex(dis(gen), dis(gen));
      nnz++;
    }
  }

  matrix.nnz = nnz;
  matrix.values.reserve(nnz);
  matrix.row_indices.reserve(nnz);
  matrix.col_offsets.resize(cols + 1, 0);

  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (temp[i][j] != Complex(0.0, 0.0)) {
        matrix.values.push_back(temp[i][j]);
        matrix.row_indices.push_back(i);
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}

SparseMatrixCCS CreateCcsFromDense(const std::vector<std::vector<Complex>>& dense) {
  int rows = static_cast<int>(dense.size());
  int cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  SparseMatrixCCS matrix(rows, cols, 0);

  matrix.col_offsets.resize(cols + 1, 0);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      if (dense[i][j] != Complex(0.0, 0.0)) {
        matrix.values.push_back(dense[i][j]);
        matrix.row_indices.push_back(i);
        matrix.nnz++;
      }
    }
    matrix.col_offsets[j + 1] = static_cast<int>(matrix.values.size());
  }
  return matrix;
}

bool RunTask(SparseMatrixCCS& m1, SparseMatrixCCS& m2, SparseMatrixCCS& result) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m1));
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(&m2));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));

  SparseMatrixMultComplexCCS task(task_data);
  if (!task.PreProcessingImpl()) {
    return false;
  }
  if (!task.ValidationImpl()) {
    return false;
  }
  task.RunImpl();
  task.PostProcessingImpl();
  return true;
}

void ExpectMatrixValuesEq(const SparseMatrixCCS& result, const SparseMatrixCCS& expected, double epsilon) {
  ASSERT_EQ(result.values.size(), expected.values.size());
  for (size_t i = 0; i < result.values.size(); i++) {
    EXPECT_NEAR(std::abs(result.values[i] - expected.values[i]), 0.0, epsilon);
  }
}

void ExpectMatrixEq(const SparseMatrixCCS& result, const SparseMatrixCCS& expected, double epsilon) {
  EXPECT_EQ(result.rows, expected.rows);
  EXPECT_EQ(result.cols, expected.cols);
  EXPECT_EQ(result.nnz, expected.nnz);
  EXPECT_EQ(result.col_offsets, expected.col_offsets);
  EXPECT_EQ(result.row_indices, expected.row_indices);
  ExpectMatrixValuesEq(result, expected, epsilon);
}
}  // namespace korneeva_e_sparse_matrix_mult_complex_ccs_seq