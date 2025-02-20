#include "seq/Sadikov_I_SparesMatrixMultiplication/include/ops_seq.hpp"

bool Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential::PreProcessingImpl() {
  auto fmatrixRowsCount = static_cast<int>(task_data->inputs_count[0]);
  auto fmatrxixColumnsCount = static_cast<int>(task_data->inputs_count[1]);
  auto smatrixRowsCount = static_cast<int>(task_data->inputs_count[2]);
  auto smatrixColumnsCount = static_cast<int>(task_data->inputs_count[3]);
  if (fmatrixRowsCount == 0 || fmatrxixColumnsCount == 0 || smatrixRowsCount == 0 || smatrixColumnsCount == 0) {
    return true;
  }
  std::vector<double> fmatrix;
  std::vector<double> smatrix;
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  for (auto i = 0; i < fmatrixRowsCount * fmatrxixColumnsCount; ++i) {
    fmatrix.emplace_back(in_ptr[i]);
  }
  m_fMatrix = MatrixToSpares(fmatrixRowsCount, fmatrxixColumnsCount, fmatrix);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  for (auto i = 0; i < smatrixRowsCount * smatrixColumnsCount; ++i) {
    smatrix.emplace_back(in_ptr2[i]);
  }
  m_sMatrix = MatrixToSpares(smatrixRowsCount, smatrixColumnsCount, smatrix);
  return true;
}

bool Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential::RunImpl() {
  m_answerMatrix = m_fMatrix * m_sMatrix;
  return true;
}

bool Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential::PostProcessingImpl() {
  auto answer = FromSparesMatrix(m_answerMatrix);
  for (auto i = 0; i < static_cast<int>(answer.size()); ++i) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = answer[i];
  }
  return true;
}
