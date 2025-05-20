#pragma once

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_radix_sort_batcher_merge_all {

class TestTaskAll : public ppc::core::Task {
 private:
  std::vector<long long int> mas_, tmp_, loc_, loc_tmp_;
  unsigned int n_, n_input_, loc_proc_lenght_;
  int effective_num_procs_;
  boost::mpi::communicator world;

 public:
  explicit TestTaskAll(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {
    if (world.rank() == 0) n_input_ = taskData->inputs_count[0];
  }
  static bool RadixUnsigned(unsigned long long *, unsigned long long *, unsigned int);
  [[nodiscard]] bool RadixSigned(unsigned int, unsigned int) const;
  static bool Countbyte(unsigned long long *, int *, unsigned int, unsigned int);
  static bool OddEvenMerge(long long int *, long long int *, const long long int *, unsigned int, unsigned int);
  bool FinalMerge();
  bool BatcherSortOMP();
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace kovalev_k_radix_sort_batcher_merge_all