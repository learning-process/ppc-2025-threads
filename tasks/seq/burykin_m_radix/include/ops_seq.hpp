#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_seq {

class RadixSequential : public ppc::core::Task {
 public:
  explicit RadixSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int rc_size_{};
};

}  // namespace burykin_m_radix_seq