// Copyright Anikin Maksim 2025
#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_shall_batcher_seq {

void ShellSort(std::vector<int>& arr);
void BatcherOddEvenMerge(std::vector<int>& arr1, std::vector<int>& arr2, std::vector<int>& output);
void ShellSortWithBatcherMerge(const std::vector<int>& input, std::vector<int>& output);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
};

}  // namespace anikin_m_shall_batcher_seq