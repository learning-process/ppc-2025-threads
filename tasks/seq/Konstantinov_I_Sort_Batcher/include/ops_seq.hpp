#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_sort_batcher {

class RadixSortBatcherSeq : public ppc::core::Task {
 public:
  explicit RadixSortBatcherSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> mas_, output_;
  void RadixSort(std::vector<double>& arr);
  void BatcherOddEvenMerge(std::vector<double>& arr, int low, int high);

};
}  // namespace konstantinov_i_sort_batcher