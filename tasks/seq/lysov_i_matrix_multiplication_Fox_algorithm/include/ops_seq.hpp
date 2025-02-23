#pragma once
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_seq {

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