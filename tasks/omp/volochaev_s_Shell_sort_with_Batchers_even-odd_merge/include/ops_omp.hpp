#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_omp {
class ShellSortOMP : public ppc::core::Task {
 public:
  explicit ShellSortOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int c_threads_;
  int mini_batch_;
  int size_;
  int n_;
  std::vector<int> array_;
  std::vector<int> mass_;

  void Merge();
  void ParallelShellSort();
  void ShellSort(int start, int finish);
  void FindThreadVariables();
  void MergeBlocks(int id_1, int block_size_1, int index_2, int block_size_2);
};

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_omp