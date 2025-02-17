#include "seq/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_seq {
void MultiplyCCS(int m, int k, int n, const std::vector<double> &a_values, const std::vector<int> &a_row_indices,
                 const std::vector<int> &a_col_ptr, const std::vector<double> &b_values,
                 const std::vector<int> &b_row_indices, const std::vector<int> &b_col_ptr,
                 std::vector<double> &c_values, std::vector<int> &c_row_indices, std::vector<int> &c_col_ptr) {
  c_values.clear();
  c_row_indices.clear();
  c_col_ptr.assign(n + 1, 0);

  std::vector<double> temp_values(m, 0.0);
  std::vector<bool> temp_used(m, false);

  for (int j = 0; j < n; ++j) {
    for (int t = b_col_ptr[j]; t < b_col_ptr[j + 1]; ++t) {
      int row_b = b_row_indices[t];
      double val_b = b_values[t];

      for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
        int row_a = a_row_indices[i];
        temp_values[row_a] += a_values[i] * val_b;
        temp_used[row_a] = true;
      }
    }

    c_col_ptr[j] = static_cast<int>(c_values.size());

    for (int i = 0; i < m; ++i) {
      if (temp_used[i]) {
        c_values.push_back(temp_values[i]);
        c_row_indices.push_back(i);
        temp_values[i] = 0.0;
        temp_used[i] = false;
      }
    }
  }
  c_col_ptr[n] = static_cast<int>(c_values.size());
}
}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_seq

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  M_ = static_cast<int>(task_data->inputs_count[0]);
  K_ = static_cast<int>(task_data->inputs_count[1]);
  N_ = static_cast<int>(task_data->inputs_count[2]);
  auto *current_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  A_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[3]);
  current_ptr = reinterpret_cast<double *>(task_data->inputs[1]);
  std::vector<double> aRowIndicesD = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[4]);
  A_row_indices_.resize(aRowIndicesD.size());
  std::ranges::transform(aRowIndicesD.begin(), aRowIndicesD.end(), A_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  std::vector<double> aColPtrD = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[5]);
  A_col_ptr_.resize(aColPtrD.size());
  std::ranges::transform(aColPtrD.begin(), aColPtrD.end(), A_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[3]);
  B_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[6]);
  current_ptr = reinterpret_cast<double *>(task_data->inputs[4]);
  std::vector<double> bRowIndicesD = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[7]);
  B_row_indices_.resize(bRowIndicesD.size());
  std::ranges::transform(bRowIndicesD.begin(), bRowIndicesD.end(), B_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  std::vector<double> bColPtrD = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[8]);
  B_col_ptr_.resize(bColPtrD.size());
  std::ranges::transform(bColPtrD.begin(), bColPtrD.end(), B_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });

  unsigned int output_size = task_data->outputs_count[0];
  C_values_ = std::vector<double>(output_size, 0);
  C_row_indices_ = std::vector<int>(output_size, 0);
  C_col_ptr_ = std::vector<int>(output_size, 0);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] != 0 && task_data->inputs_count[1] != 0 && task_data->inputs_count[2] != 0;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::RunImpl() {
  // Multiply matrices
  MultiplyCCS(M_, K_, N_, A_values_, A_row_indices_, A_col_ptr_, B_values_, B_row_indices_, B_col_ptr_, C_values_,
              C_row_indices_, C_col_ptr_);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_seq::TestTaskSequential::PostProcessingImpl() {
  std::vector<double> c_row_indices_d(C_row_indices_.size());
  std::vector<double> c_col_ptr_d(C_col_ptr_.size());
  std::ranges::transform(C_row_indices_.begin(), C_row_indices_.end(), c_row_indices_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  std::ranges::transform(C_col_ptr_.begin(), C_col_ptr_.end(), c_col_ptr_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  for (size_t i = 0; i < C_values_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = C_values_[i];
  }
  for (size_t i = 0; i < c_row_indices_d.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[1])[i] = c_row_indices_d[i];
  }
  for (size_t i = 0; i < c_col_ptr_d.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[2])[i] = c_col_ptr_d[i];
  }
  return true;
}
