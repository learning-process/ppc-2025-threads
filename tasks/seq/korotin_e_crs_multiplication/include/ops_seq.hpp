#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_crs_multiplication_seq {

class CrsMultiplicationSeqrential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_val_, B_val_, output_val_;
  std::vector<unsigned int> A_col_, A_rI_, B_col_, B_rI_, output_col_, output_rI_;
  unsigned int A_N_, A_Nz_, B_N_, B_Nz_;
};

}  // korotin_e_crs_multiplication_seq
