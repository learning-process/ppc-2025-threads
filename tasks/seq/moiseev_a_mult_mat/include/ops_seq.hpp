#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace moiseev_a_mult_mat_seq {

class MultMatSequential : public ppc::core::Task {
public:
  explicit MultMatSequential(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

private:
  std::vector<double> matrix_A_, matrix_B_, matrix_C_;
  int matrix_size_{};
  int num_blocks_{};
  int block_size_{};
};

} // namespace moiseev_a_mult_mat_seq