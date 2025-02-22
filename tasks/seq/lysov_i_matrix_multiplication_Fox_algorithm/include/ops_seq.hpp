#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_Fox_algorithm_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C;
  std::size_t N;
  std::size_t block_size;
};

}  // namespace lysov_i_matrix_multiplication_Fox_algorithm_seq