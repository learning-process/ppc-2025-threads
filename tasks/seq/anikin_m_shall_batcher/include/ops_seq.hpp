// Copyright Anikin Maksim 2025
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_shall_batcher_seq {

void shellSort(std::vector<int>& arr);
void batcherOddEvenMerge(std::vector<int>& arr1, std::vector<int>& arr2, std::vector<int>& output);
void shellSortWithBatcherMerge(const std::vector<int>& input, std::vector<int>& output);

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

}