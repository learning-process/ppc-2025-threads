#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_bin_labeling {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void dfs(int x, int y, int cnt);

 private:
  int width_, height_;
  std::vector<int> input_{};
  std::vector<int> res_{};
};

}  // namespace shkurinskaya_e_bin_labeling