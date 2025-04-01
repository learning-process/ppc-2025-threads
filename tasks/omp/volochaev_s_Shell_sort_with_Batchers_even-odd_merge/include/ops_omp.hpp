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
  int thread_num_;
  int thread_id_;
  int dim_size_;
  void InitializeParallelSections();
  void MergeBlocks(std::vector<int>& p_data, int index_1, int block_size_1, int index_2, int block_size_2);
  void ShellSort(std::vector<int>& arr, int start, int finish);
  bool IsSorted(std::vector<int>& p_data, int size);
  int GrayCode(int ring_id, int dim_size);
  int ReverseGrayCode(int cube_id, int dim_size);
  void SetBlockPairs(int* block_pairs, int iter);
  int FindMyPair(int* block_pairs, int thread_id, int iter);
  void ParallelShellSort(std::vector<int>& p_data, int dize);

  std::vector<int> array_;
};

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_omp