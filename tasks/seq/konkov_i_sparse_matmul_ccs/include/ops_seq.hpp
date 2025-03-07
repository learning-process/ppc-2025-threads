#pragma once

#include <cstdint>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_sparse_matmul_ccs {

class SparseMatmulTask : public ppc::core::Task {
 public:
  explicit SparseMatmulTask(ppc::core::TaskDataPtr task_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> A_values, B_values, C_values;
  std::vector<int> A_columns, B_columns, C_columns;
  int rowsA, colsA, rowsB, colsB;
};

}  // namespace konkov_i_sparse_matmul_ccs
