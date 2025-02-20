#pragma once

#include <cmath>
#include <utility>

#include "core/task/include/task.hpp"
#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

namespace sadikov_i_sparse_matrix_multiplication_task_seq {

class CCSMatrixSequential : public ppc::core::Task {
  SparesMatrix m_fMatrix_;
  SparesMatrix m_sMatrix_;
  SparesMatrix m_answerMatrix_;

 public:
  explicit CCSMatrixSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace Sadikov_I_SparseMatrixMultiplication_task_seq