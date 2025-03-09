// Copyright 2025 Kavtorev Dmitry
#include "seq/kavtorev_d_dense_matrix_cannon/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

std::vector<double> kavtorev_d_dense_matrix_cannon_seq::cannonMatrixMultiplication(const std::vector<double>& A,
                                                                                   const std::vector<double>& B, int n,
                                                                                   int m) {
  int SizeBlock = std::min(n, m);

  std::vector<double> mtrx_C(n * m, 0.0);

  if (n == 0 || m == 0) {
    return std::vector<double>();
  }

  for (int i = 0; i < n; i += SizeBlock) {
    for (int j = 0; j < m; j += SizeBlock) {
      for (int k = 0; k < m; k += SizeBlock) {
        int i_end = std::min(i + SizeBlock, n);
        int j_end = std::min(j + SizeBlock, m);
        int k_end = std::min(k + SizeBlock, m);

        for (int ii = i; ii < i_end; ++ii) {
          for (int kk = k; kk < k_end; ++kk) {
            double A_ik = A[ii * m + kk];
            for (int jj = j; jj < j_end; ++jj) {
              mtrx_C[ii * m + jj] += A_ik * B[kk * m + jj];
            }
          }
        }
      }
    }
  }

  return mtrx_C;
}

std::vector<double> kavtorev_d_dense_matrix_cannon_seq::multiplyMatrix(const std::vector<double>& A,
                                                                       const std::vector<double>& B, int rows_A,
                                                                       int col_B) {
  int col_A = rows_A;
  std::vector<double> mtrx_C(rows_A * col_B, 0.0);

  if (rows_A == 0 || col_B == 0) {
    return std::vector<double>();
  }

  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < col_B; ++j) {
      for (int k = 0; k < col_A; ++k) {
        mtrx_C[i * col_B + j] += A[i * col_A + k] * B[k * col_B + j];
      }
    }
  }
  return mtrx_C;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::PreProcessingImpl() {
  A = std::vector<double>(task_data->inputs_count[0]);
  B = std::vector<double>(task_data->inputs_count[1]);
  n = *reinterpret_cast<int*>(task_data->inputs[2]);
  m = *reinterpret_cast<int*>(task_data->inputs[3]);

  auto* tmp_ptr_A = reinterpret_cast<double*>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    A[i] = tmp_ptr_A[i];
  }

  auto* tmp_ptr_B = reinterpret_cast<double*>(task_data->inputs[1]);
  for (size_t i = 0; i < task_data->inputs_count[1]; i++) {
    B[i] = tmp_ptr_B[i];
  }
  return true;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         task_data->inputs_count[1] == task_data->outputs_count[0];
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::RunImpl() {
  res = cannonMatrixMultiplication(A, B, n, m);
  return true;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(res.begin(), res.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}