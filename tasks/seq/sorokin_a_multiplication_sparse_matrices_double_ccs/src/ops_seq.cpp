#include "seq/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

void printCCS(const std::vector<double> &values, const std::vector<int> &row_indices, const std::vector<int> &col_ptr) {
  std::cout << "Values: ";
  for (double v : values) std::cout << v << " ";
  std::cout << "\nRow indices: ";
  for (int r : row_indices) std::cout << r << " ";
  std::cout << "\nColumn pointers: ";
  for (int c : col_ptr) std::cout << c << " ";
  std::cout << "\n";
}

void multiplyCCS(int M, int K, int N, const std::vector<double> &A_values, const std::vector<int> &A_row_indices,
                 const std::vector<int> &A_col_ptr, const std::vector<double> &B_values,
                 const std::vector<int> &B_row_indices, const std::vector<int> &B_col_ptr,
                 std::vector<double> &C_values, std::vector<int> &C_row_indices, std::vector<int> &C_col_ptr) {
  C_values.clear();
  C_row_indices.clear();
  C_col_ptr.assign(N + 1, 0);

  std::vector<double> temp_values(M, 0.0);
  std::vector<bool> temp_used(M, false);

  for (int j = 0; j < N; ++j) {
    for (int k = B_col_ptr[j]; k < B_col_ptr[j + 1]; ++k) {
      int row_B = B_row_indices[k];
      double val_B = B_values[k];

      for (int i = A_col_ptr[row_B]; i < A_col_ptr[row_B + 1]; ++i) {
        int row_A = A_row_indices[i];
        temp_values[row_A] += A_values[i] * val_B;
        temp_used[row_A] = true;
      }
    }

    C_col_ptr[j] = C_values.size();

    for (int i = 0; i < M; ++i) {
      if (temp_used[i]) {
        C_values.push_back(temp_values[i]);
        C_row_indices.push_back(i);
        temp_values[i] = 0.0;
        temp_used[i] = false;
      }
    }
  }
  C_col_ptr[N] = C_values.size();
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  M = task_data->inputs_count[0];
  K = task_data->inputs_count[1];
  N = task_data->inputs_count[2];
  auto *current_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  A_values = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[3]);
  current_ptr = reinterpret_cast<double *>(task_data->inputs[1]);
  std::vector<double> A_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[4]);
  A_row_indices.resize(A_row_indices_d.size());
  std::transform(A_row_indices_d.begin(), A_row_indices_d.end(), A_row_indices.begin(),
                 [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  std::vector<double> A_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[5]);
  A_col_ptr.resize(A_col_ptr_d.size());
  std::transform(A_col_ptr_d.begin(), A_col_ptr_d.end(), A_col_ptr.begin(),
                 [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[3]);
  B_values = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[6]);
  current_ptr = reinterpret_cast<double *>(task_data->inputs[4]);
  std::vector<double> B_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[7]);
  B_row_indices.resize(B_row_indices_d.size());
  std::transform(B_row_indices_d.begin(), B_row_indices_d.end(), B_row_indices.begin(),
                 [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  std::vector<double> B_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[8]);
  B_col_ptr.resize(B_col_ptr_d.size());
  std::transform(B_col_ptr_d.begin(), B_col_ptr_d.end(), B_col_ptr.begin(),
                 [](double x) { return static_cast<int>(x); });

  unsigned int output_size = task_data->outputs_count[0];
  C_values = std::vector<double>(output_size, 0);
  C_row_indices = std::vector<int>(output_size, 0);
  C_col_ptr = std::vector<int>(output_size, 0);

  // printCCS(A_values, A_row_indices, A_col_ptr);
  // printCCS(B_values, B_row_indices, B_col_ptr);

  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] != 0 && task_data->inputs_count[1] != 0 && task_data->inputs_count[2] != 0;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::RunImpl() {
  // Multiply matrices
  multiplyCCS(M, K, N, A_values, A_row_indices, A_col_ptr, B_values, B_row_indices, B_col_ptr, C_values, C_row_indices,
              C_col_ptr);
  // printCCS(C_values, C_row_indices, C_col_ptr);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::PostProcessingImpl() {
  std::vector<double> C_row_indices_d;
  C_row_indices_d.resize(C_row_indices.size());
  std::vector<double> C_col_ptr_d;
  C_col_ptr_d.resize(C_col_ptr.size());
  std::transform(C_row_indices.begin(), C_row_indices.end(), C_row_indices_d.begin(),
                 [](int x) { return static_cast<double>(x); });
  std::transform(C_col_ptr.begin(), C_col_ptr.end(), C_col_ptr_d.begin(), [](int x) { return static_cast<double>(x); });
  for (size_t i = 0; i < C_values.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = C_values[i];
  }
  for (size_t i = 0; i < C_row_indices_d.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[1])[i] = C_row_indices_d[i];
  }
  for (size_t i = 0; i < C_col_ptr_d.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[2])[i] = C_col_ptr_d[i];
  }
  return true;
}
