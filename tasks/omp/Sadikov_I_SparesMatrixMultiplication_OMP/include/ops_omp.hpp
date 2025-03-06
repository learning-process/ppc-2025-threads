#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/Sadikov_I_SparesMatrixMultiplication_OMP/include/SparesMatrix.hpp"
namespace sadikov_i_sparse_matrix_multiplication_task_omp {

std::vector<double> GetRandomMatrix(int size);
class CCSMatrixOMP : public ppc::core::Task {
  SparesMatrix m_fMatrix_;
  SparesMatrix m_sMatrix_;
  SparesMatrix m_answerMatrix_;

 public:
  explicit CCSMatrixOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sadikov_i_sparse_matrix_multiplication_task_omp