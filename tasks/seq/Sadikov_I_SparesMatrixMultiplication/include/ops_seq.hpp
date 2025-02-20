#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

namespace Sadikov_I_SparseMatrixMultiplication_task_seq {

class CCSMatrixSequential : public ppc::core::Task {
  SparesMatrix m_fMatrix;
  SparesMatrix m_sMatrix;
  SparesMatrix m_answerMatrix;

 public:
  explicit CCSMatrixSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace Sadikov_I_SparseMatrixMultiplication_task_seq