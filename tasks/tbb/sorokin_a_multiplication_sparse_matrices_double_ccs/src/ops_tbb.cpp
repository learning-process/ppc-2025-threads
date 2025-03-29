#include "tbb/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_tbb {

void MultiplyCCS(const std::vector<double>& a_values, const std::vector<int>& a_row_indices, int m,
                 const std::vector<int>& a_col_ptr, const std::vector<double>& b_values,
                 const std::vector<int>& b_row_indices, int k, const std::vector<int>& b_col_ptr,
                 std::vector<double>& c_values, std::vector<int>& c_row_indices, int n, std::vector<int>& c_col_ptr) {
  if (static_cast<int>(a_values.size()) > m * k || static_cast<int>(b_values.size()) > k * n) {
    throw std::invalid_argument("Invalid matrix values size");
  }

  c_col_ptr.assign(n + 1, 0);
  std::vector<int> col_sizes(n, 0);

  tbb::parallel_for(tbb::blocked_range<int>(0, n), [&](const auto& range) {
    for (int j = range.begin(); j < range.end(); ++j) {
      std::vector<bool> row_marks(m, false);

      for (int b_idx = b_col_ptr[j]; b_idx < b_col_ptr[j + 1]; ++b_idx) {
        const int row_b = b_row_indices[b_idx];

        for (int a_idx = a_col_ptr[row_b]; a_idx < a_col_ptr[row_b + 1]; ++a_idx) {
          const int row_a = a_row_indices[a_idx];
          if (!row_marks[row_a]) {
            row_marks[row_a] = true;
            col_sizes[j]++;
          }
        }
      }
    }
  });

  for (int j = 0; j < n; ++j) {
    c_col_ptr[j + 1] = c_col_ptr[j] + col_sizes[j];
  }

  c_values.resize(c_col_ptr.back());
  c_row_indices.resize(c_col_ptr.back());

  tbb::parallel_for(tbb::blocked_range<int>(0, n), [&](const auto& range) {
    for (int j = range.begin(); j < range.end(); ++j) {
      std::vector<double> temp_values(m, 0.0);
      std::vector<bool> temp_used(m, false);
      int pos = c_col_ptr[j];

      for (int b_idx = b_col_ptr[j]; b_idx < b_col_ptr[j + 1]; ++b_idx) {
        const int row_b = b_row_indices[b_idx];
        const double b_val = b_values[b_idx];

        for (int a_idx = a_col_ptr[row_b]; a_idx < a_col_ptr[row_b + 1]; ++a_idx) {
          const int row_a = a_row_indices[a_idx];
          temp_values[row_a] += a_values[a_idx] * b_val;
          temp_used[row_a] = true;
        }
      }

      for (int i = 0; i < m; ++i) {
        if (temp_used[i]) {
          c_row_indices[pos] = i;
          c_values[pos] = temp_values[i];
          pos++;
        }
      }
    }
  });
}

}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_tbb

bool sorokin_a_multiplication_sparse_matrices_double_ccs_tbb::TestTaskTBB::PreProcessingImpl() {
  // Init value for input and output
  M_ = static_cast<int>(task_data->inputs_count[0]);
  K_ = static_cast<int>(task_data->inputs_count[1]);
  N_ = static_cast<int>(task_data->inputs_count[2]);
  auto *current_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  A_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[3]);
  current_ptr = reinterpret_cast<double *>(task_data->inputs[1]);
  std::vector<double> a_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[4]);
  A_row_indices_.resize(a_row_indices_d.size());
  std::ranges::transform(a_row_indices_d.begin(), a_row_indices_d.end(), A_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  std::vector<double> a_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[5]);
  A_col_ptr_.resize(a_col_ptr_d.size());
  std::ranges::transform(a_col_ptr_d.begin(), a_col_ptr_d.end(), A_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[3]);
  B_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[6]);
  current_ptr = reinterpret_cast<double *>(task_data->inputs[4]);
  std::vector<double> b_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[7]);
  B_row_indices_.resize(b_row_indices_d.size());
  std::ranges::transform(b_row_indices_d.begin(), b_row_indices_d.end(), B_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  std::vector<double> b_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[8]);
  B_col_ptr_.resize(b_col_ptr_d.size());
  std::ranges::transform(b_col_ptr_d.begin(), b_col_ptr_d.end(), B_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_tbb::TestTaskTBB::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_tbb::TestTaskTBB::RunImpl() {
  MultiplyCCS(A_values_, A_row_indices_, M_, A_col_ptr_, B_values_, B_row_indices_, K_, B_col_ptr_, C_values_,
              C_row_indices_, N_, C_col_ptr_);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_tbb::TestTaskTBB::PostProcessingImpl() {
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
