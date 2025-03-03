#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korovin_n_qsort_batcher_omp {

struct BlockRange {
  int start;
  int length;
};

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  static int GetRandomIndex(int low, int high);
  static void QuickSort(std::vector<int>& arr, int low, int high, int depth = 0);
  static bool InPlaceMerge(std::vector<int>& arr, const BlockRange& a, const BlockRange& b, std::vector<int>& buffer);
  static std::vector<BlockRange> PartitionBlocks(int n, int p);
  void OddEvenMerge(std::vector<BlockRange>& blocks);
};

}  // namespace korovin_n_qsort_batcher_omp
