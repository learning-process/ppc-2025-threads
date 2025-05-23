#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_hoare_batcher_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  void QuickSort(std::vector<int>& arr, int low, int high);
  static int Partition(std::vector<int>& arr, int low, int high);
  static void BatcherMerge(std::vector<int>& arr, int left, int mid, int right);
};

}  // namespace dudchenko_o_hoare_batcher_seq