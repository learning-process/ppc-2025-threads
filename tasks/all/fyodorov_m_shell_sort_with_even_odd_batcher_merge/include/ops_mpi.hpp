#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {
class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  [[nodiscard]] const std::vector<int>& GetInternalOutput() const { return output_; }

 private:
  static void ShellSort(std::vector<int>& arr);
  static void BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);
  static void PrepareScatterGather(int n, int size, std::vector<int>& sendcounts, std::vector<int>& displs);
  static std::vector<std::vector<int>> SplitGatheredToBlocks(const std::vector<int>& gathered,
                                                             const std::vector<int>& sendcounts);
  static std::vector<int> MergeBlocks(const std::vector<std::vector<int>>& blocks);
  static void BroadcastOutput(boost::mpi::communicator& world, int rank, int size, std::vector<int>& output);
  static void LocalSort(std::vector<int>& local_data, int rank);
  static void PrintFirstN(const std::string& label, const std::vector<int>& data, int n = 10);
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<int> local_input_;
};
}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi