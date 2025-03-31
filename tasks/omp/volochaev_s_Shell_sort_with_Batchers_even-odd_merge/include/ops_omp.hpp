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
  int threadnum_;
  int threadid_;
  int dimsize_;
  void InitializeParallelSections();
  void MergeBlocks(std::vector<int>& pData, int Index1, int BlockSize1, int Index2, int BlockSize2);
  void ShellSort(std::vector<int>& arr, int start, int finish);
  bool IsSorted(std::vector<int>& pData, int size);
  int GrayCode(int RingID, int DimSize);
  int ReverseGrayCode(int CubeID, int DimSize);
  void SetBlockPairs(int* BlockPairs, int Iter);
  int FindMyPair(int* BlockPairs, int ThreadID, int Iter);
  void ParallelShellSort(std::vector<int>& pData, int Size);

  std::vector<int> array_;
};

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_omp