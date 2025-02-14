#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_seq {
std::vector<int> ShellSort(const std::vector<int>& input_array);
std::vector<int> GenerateRandomVector(int size, int max_value, int min_value);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, result_;
};
}  // namespace sotskov_a_shell_sorting_with_simple_merging_seq