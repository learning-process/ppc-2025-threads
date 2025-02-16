#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_integral_simpson_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int rc_size_{};
};

}  // namespace kolokolova_d_integral_simpson_method_seq