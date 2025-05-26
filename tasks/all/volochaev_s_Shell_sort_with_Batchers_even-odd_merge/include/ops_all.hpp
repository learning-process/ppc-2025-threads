#pragma once

#include <stddef.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all {

class ShellSortALL : public ppc::core::Task {
 private:
  boost::mpi::communicator world_;
  std::vector<long long int> loc_, loc_tmp_;
  std::vector<long long int> mas_;
  size_t loc_proc_lenght_;
  size_t effective_num_procs_;
  size_t n_input_, n_;

 public:
  explicit ShellSortALL(const std::shared_ptr<ppc::core::TaskData>& task_data) : Task(task_data) {
    if (world_.rank() == 0) {
      n_input_ = task_data->inputs_count[0];
    };
  }

  static std::vector<size_t> GenerateGaps(size_t size);
  void ShellSort(std::vector<long long>& arr, size_t start, size_t size);
  bool BatcherSort();
  static bool OddEvenMergeSTL(std::vector<long long>& tmp, const std::vector<long long>& left,
                              const std::vector<long long>& right);
  static bool FinalMergeSTL(std::vector<long long>& loc, std::vector<long long>& loc_tmp);
  bool OddEvenMergeMPI(unsigned int len);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all