#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq {
class ShellSortSequential : public ppc::core::Task {
 public:
  explicit ShellSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> array_;
  void shellSort(std::vector<int>& arr);
};

}  // namespace volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq