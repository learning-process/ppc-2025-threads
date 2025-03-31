#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_omp {
void TrivialMatrixMultiplication(const std::vector<double> &matrix_a, const std::vector<double> &matrix_b,
                                 std::vector<double> &result_matrix, size_t matrix_size);
std::vector<double> GetRandomMatrix(size_t size);
void ProcessBlock(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, std::size_t i,
                  std::size_t j, std::size_t a_block_row, std::size_t block_size, std::size_t n);

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> c_;
  std::size_t n_;
  std::size_t block_size_;
};

}  // namespace lysov_i_matrix_multiplication_fox_algorithm_omp