#include "tbb/sadikov_I_SparseMatMul_TBB/include/ops_tbb.hpp"

#include <random>
#include <vector>
#include <iostream>
#include "core/util/include/util.hpp"

bool sadikov_i_sparse_matrix_multiplication_task_tbb::CCSMatrixTBB::PreProcessingImpl() {
  auto fmatrix_rows_count = static_cast<int>(task_data->inputs_count[0]);
  auto fmatrxix_columns_count = static_cast<int>(task_data->inputs_count[1]);
  auto smatrix_rows_count = static_cast<int>(task_data->inputs_count[2]);
  auto smatrix_columns_count = static_cast<int>(task_data->inputs_count[3]);
  if (fmatrix_rows_count == 0 || fmatrxix_columns_count == 0 || smatrix_rows_count == 0 || smatrix_columns_count == 0) {
    return true;
  }
  std::vector<double> fmatrix;
  fmatrix.reserve(fmatrix_rows_count * fmatrxix_columns_count);
  std::vector<double> smatrix;
  smatrix.reserve(smatrix_rows_count * smatrix_columns_count);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < fmatrix_rows_count * fmatrxix_columns_count; ++i) {
    fmatrix.emplace_back(in_ptr[i]);
  }
  m_fMatrix_ = SparseMatrix::MatrixToSparse(fmatrix_rows_count, fmatrxix_columns_count, fmatrix);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  for (int i = 0; i < smatrix_rows_count * smatrix_columns_count; ++i) {
    smatrix.emplace_back(in_ptr2[i]);
  }
  m_sMatrix_ = SparseMatrix::MatrixToSparse(smatrix_rows_count, smatrix_columns_count, smatrix);
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_tbb::CCSMatrixTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0];
}

bool sadikov_i_sparse_matrix_multiplication_task_tbb::CCSMatrixTBB::RunImpl() {
  m_answerMatrix_ = m_fMatrix_ * m_sMatrix_;
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_tbb::CCSMatrixTBB::PostProcessingImpl() {
  auto answer = FromSparseMatrix(m_answerMatrix_);
  for (size_t i = 0; i < answer.size(); ++i) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = answer[i];
  }
  return true;
}

std::vector<double> sadikov_i_sparse_matrix_multiplication_task_tbb::GetRandomMatrix(int size) {
  std::vector<double> data(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<double>(gen() % 500);
    if (data[i] > 250.0) {
      data[i] = 0.0;
    }
  }
  return data;
}