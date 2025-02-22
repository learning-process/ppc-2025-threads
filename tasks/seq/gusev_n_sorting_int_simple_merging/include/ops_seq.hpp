#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_sorting_int_simple_merging_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void CountingSort(std::vector<int>& arr, int exp);
  void RadixSort(std::vector<int>& arr);
  void RadixSortForNonNegative(std::vector<int>& arr);
 private:
  std::vector<int> input_, output_;
  int rc_size_{};
};

}  // namespace gusev_n_sorting_int_simple_merging_seq