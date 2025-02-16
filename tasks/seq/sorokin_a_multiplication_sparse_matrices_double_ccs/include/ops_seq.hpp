#pragma once

#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_seq {

// Объявление функции printCCS
void printCCS(const std::vector<double> &values, const std::vector<int> &row_indices, const std::vector<int> &col_ptr);

void multiplyCCS(int M, int K, int N, const std::vector<double> &A_values, const std::vector<int> &A_row_indices,
                 const std::vector<int> &A_col_ptr, const std::vector<double> &B_values,
                 const std::vector<int> &B_row_indices, const std::vector<int> &B_col_ptr,
                 std::vector<double> &C_values, std::vector<int> &C_row_indices, std::vector<int> &C_col_ptr);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int M;
  int K;
  int N;
  std::vector<double> A_values;
  std::vector<int> A_row_indices;
  std::vector<int> A_col_ptr;
  std::vector<double> B_values;
  std::vector<int> B_row_indices;
  std::vector<int> B_col_ptr;
  std::vector<double> C_values;
  std::vector<int> C_row_indices;
  std::vector<int> C_col_ptr;
};

}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_seq