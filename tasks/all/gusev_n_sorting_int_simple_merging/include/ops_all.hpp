#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_sorting_int_simple_merging_all {

class SortingIntSimpleMergingALL : public ppc::core::Task {
 public:
  explicit SortingIntSimpleMergingALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void CountingSort(std::vector<int>& arr, int exp);
  static void RadixSortForNonNegative(std::vector<int>& arr);
  static void RadixSort(std::vector<int>& arr);

  static void SplitBySign(const std::vector<int>& arr, std::vector<int>& negatives, std::vector<int>& positives);
  static void MergeResults(std::vector<int>& arr, const std::vector<int>& negatives, const std::vector<int>& positives);
  static std::vector<std::vector<int>> DistributeArray(const std::vector<int>& arr, int num_procs);
  static std::vector<int> MergeSortedArrays(const std::vector<std::vector<int>>& arrays);

  std::vector<int> input_, output_;
};

}  // namespace gusev_n_sorting_int_simple_merging_all
