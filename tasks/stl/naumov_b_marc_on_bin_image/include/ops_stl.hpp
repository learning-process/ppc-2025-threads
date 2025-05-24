#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_marc_on_bin_image_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int rc_size_{};
};

}  // namespace nesterov_a_test_task_stl
