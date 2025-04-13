#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all {

class ShellSortALL : public ppc::core::Task {
 public:
  explicit ShellSortALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int rank_{0};
  int world_size_{1};
  std::vector<int> local_data_;

  int c_threads_{1};
  int mini_batch_{0};
  int size_{0};
  int n_{0};
  int n_local_{0};

  std::vector<int> array_;
  std::vector<int> mass_;

  void DistributeData();
  void GatherAndMerge();

  void ParallelShellSortLocal();
  void ShellSort(int start);
  void MergeLocal();
  void MergeBlocks(int id_l, int id_r, int len);
  void LastMerge();
  void FindThreadVariables();
};

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all