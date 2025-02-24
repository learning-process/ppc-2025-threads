#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_seq {

void ProcessBlock(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, std::size_t i,
                  std::size_t j, std::size_t a_block_row, std::size_t block_size, std::size_t n);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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

}  // namespace lysov_i_matrix_multiplication_fox_algorithm_seq