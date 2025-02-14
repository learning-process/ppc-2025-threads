#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_seq {
struct RandomVectorParams {
  int size;
  int min_value;
  int max_value;
};
void ShellSortWithSimpleMerging(std::vector<int>& arr);
void SimpleMerge(std::vector<int>& arr, std::size_t left, std::size_t mid, std::size_t right);
std::vector<int> GenerateRandomVector(const RandomVectorParams& params);
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