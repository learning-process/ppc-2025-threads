// Copyright 2025 Kavtorev Dmitry
#pragma once

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace kavtorev_d_dense_matrix_cannon_seq {
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> res;
  int n = 0, m = 0;
};

std::vector<double> multiplyMatrix(const std::vector<double>& A, const std::vector<double>& B, int rows_A, int col_B);
std::vector<double> cannonMatrixMultiplication(const std::vector<double>& A, const std::vector<double>& B, int n,
                                               int m);

}  // namespace kavtorev_d_dense_matrix_cannon_seq