#pragma once

#include <oneapi/tbb/task_group.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_hoare_sort_simple_merge_mpi {

void SeqHoaraSort(std::vector<double>::iterator first, std::vector<double>::iterator last);
void HoaraSort(std::vector<double>::iterator first, std::vector<double>::iterator last);
void MergeTwoParts(std::vector<double>::iterator first, std::vector<double>::iterator last);
void MergeUnequalTwoParts(std::vector<double>::iterator first, std::vector<double>::iterator mid,
                                                               std::vector<double>::iterator last);

class HoareSortTaskSequential : public ppc::core::Task {
 public:
  explicit HoareSortTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_array_A_;
  size_t dimension_;
  size_t min_chunk_size_;
  size_t chunk_count_;
};
class HoareSortTaskMPI : public ppc::core::Task {
 public:
  explicit HoareSortTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_array_A_;
  size_t dimension_;
  size_t min_chunk_size_;
  size_t rest_;
  size_t chunk_count_;
  boost::mpi::communicator world;
};
}  // namespace deryabin_m_hoare_sort_simple_merge_mpi
