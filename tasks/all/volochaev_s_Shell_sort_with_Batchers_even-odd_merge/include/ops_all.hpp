#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
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
  boost::mpi::communicator world_;
  int rank_{0};
  int world_size_{1};

  std::vector<int> array_stl_;
  std::vector<int> mass_stl_;
  int n_stl_;
  int n_mpi_;
  int c_threads_stl_{1};
  int c_threads_mpi_{1};

  int mini_batch_stl_{0};
  int mini_batch_mpi_{0};

  int size_{0};
  int n_local_{0};

  std::vector<int> sizes_;
  std::vector<int> array_mpi_;
  std::vector<int> mass_mpi_;

  void DistributeData();
  void GatherAndMerge();

  static std::vector<int> Merge(std::vector<int>& v1, std::vector<int>& v2);
  void ParallelShellSort();
  void ShellSort(int start);
  void MergeBlocksSTL(int id_l, int id_r, int len);
  void MergeSTL();
  void LastMergeSTL();
  void FindThreadVariablesSTL();
};

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all