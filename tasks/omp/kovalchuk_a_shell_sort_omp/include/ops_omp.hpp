#pragma once
#include "core/task/include/task.hpp"
#include <vector>

namespace kovalchuk_a_shell_sort_omp {

class ShellSortOMP : public ppc::core::Task {
public:
  explicit ShellSortOMP(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

private:
  std::vector<int> input_;
  void ShellSort();
};

} // namespace kovalchuk_a_shell_sort_omp