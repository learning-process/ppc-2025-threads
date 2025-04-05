#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nikolaev_r_hoare_sort_simple_merge_all {

class HoareSortSimpleMergeALL : public ppc::core::Task {
 public:
  explicit HoareSortSimpleMergeALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void QuickSort(std::vector<double> &vec, size_t low, size_t high);

 private:
  std::vector<double> vect_;
  size_t vect_size_{};

  boost::mpi::communicator world_;
};

}  // namespace nikolaev_r_hoare_sort_simple_merge_all