#include "seq/sarafanov_m_CanonMatMul/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
bool sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential::PreProcessingImpl() {
  a_matrix_.ClearMatrix();
  b_matrix_.ClearMatrix();
  c_matrix_.ClearMatrix();
  int size = static_cast<int>(task_data->inputs_count[0]);
  std::vector<double> matrix_a(size);
  auto *in = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < size; ++i) {
    matrix_a[i] = in[i];
  }
  a_matrix_.SetBaseMatrix(matrix_a);
  a_matrix_.PreRoutine(MatrixType::RowMatrix);
  std::vector<double> matrix_b(size);
  auto *in2 = reinterpret_cast<double *>(task_data->inputs[1]);
  for (int i = 0; i < size; ++i) {
    matrix_b[i] = in2[i];
  }
  b_matrix_.SetBaseMatrix(matrix_b);
  b_matrix_.PreRoutine(MatrixType::ColumnMatrix);
  return true;
}

bool sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential::ValidationImpl() {
  double trunced = std::trunc(std::sqrt(static_cast<double>(task_data->inputs_count[0])));
  return std::sqrt(static_cast<double>(task_data->inputs_count[0])) - trunced < kInaccuracy &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential::RunImpl() {
  std::vector<CanonMatrix> mul_results(a_matrix_.GetSize());
  for (size_t i = 0; i < a_matrix_.GetSize(); ++i) {
    mul_results[i] = a_matrix_.MultiplicateMatrix(b_matrix_, i);
  }
  for (auto i = 0; i < static_cast<int>(mul_results.size()); ++i) {
    c_matrix_ = c_matrix_ + mul_results[i];
  }
  c_matrix_.Transpose();
  return true;
}

bool sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential::PostProcessingImpl() {
  auto matrix = c_matrix_.GetMatrix();
  for (size_t i = 0; i < matrix.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = matrix[i];
  }
  return true;
}

std::vector<double> sarafanov_m_canon_mat_mul_seq::GenerateRandomData(int size) {
  std::vector<double> matrix(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (auto i = 0; i < size; ++i) {
    matrix[i] = static_cast<double>(gen() % 300);
  }
  return matrix;
}

std::vector<double> sarafanov_m_canon_mat_mul_seq::GenerateSingleMatrix(int size) {
  std::vector<double> matrix(size, 0.0);
  int sqrt_size = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt_size; ++i) {
    for (int j = 0; j < sqrt_size; ++j) {
      if (i == j) {
        matrix[(sqrt_size * i) + j] = 1.0;
      }
    }
  }
  return matrix;
}