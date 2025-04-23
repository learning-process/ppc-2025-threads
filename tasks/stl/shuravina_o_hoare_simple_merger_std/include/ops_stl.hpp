#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessing() override;
  bool Validation() override;
  bool Run() override;
  bool PostProcessing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace shuravina_o_hoare_simple_merger_stl