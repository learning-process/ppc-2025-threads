#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_histogram_linear_stretching_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> img_;
};

}  // namespace varfolomeev_g_histogram_linear_stretching_seq