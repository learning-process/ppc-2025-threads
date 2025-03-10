#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  if (A_values.empty() || B_values.empty()) {
    std::cerr << "Error: Empty matrix data\n";
    return false;
  }
  if (colsA != rowsB) {
    std::cerr << "Error: Matrices dimensions mismatch\n";
    return false;
  }
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() { return true; }

bool SparseMatmulTask::RunImpl() {
  C_values.resize(rowsA * colsB, 0.0);

  for (size_t i = 0; i < A_values.size(); ++i) {
    for (size_t j = 0; j < B_values.size(); ++j) {
      C_values[i] += A_values[i] * B_values[j];
    }
  }
  return true;
}

bool SparseMatmulTask::PostProcessingImpl() {
  /*std::cout << "Result matrix C:\n";
  for (const auto& value : C_values) {
    std::cout << value << " ";
  }
  std::cout << '\n';*/
  return true;
}

}  // namespace konkov_i_sparse_matmul_ccs
