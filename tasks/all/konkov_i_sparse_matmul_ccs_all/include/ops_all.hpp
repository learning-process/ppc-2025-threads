#pragma once
#include <mpi.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs_all {

class SparseMatmulTask : public ppc::core::Task {
 public:
  explicit SparseMatmulTask(ppc::core::TaskDataPtr task_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> A_values, B_values, C_values;
  std::vector<int> A_row_indices, B_row_indices, C_row_indices;
  std::vector<int> A_col_ptr, B_col_ptr, C_col_ptr;
  int rowsA, colsA, rowsB, colsB;

 private:
  void ProcessColumns(int start_col, int end_col);
  void GatherResults();
  int world_rank, world_size;
};

}  // namespace konkov_i_sparse_matmul_ccs_all