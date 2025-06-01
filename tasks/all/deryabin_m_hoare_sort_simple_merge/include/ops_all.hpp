#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_hoare_sort_simple_merge_mpi {
pivot_calculation(std::vector<double>::iterator first, std::vector<double>::iterator last);
forwarding_and_merging(size_t world_rank, size_t step, bool is_even, size_t min_chunk_size_, size_t chunk_count_,
                       size_t rest_, boost::mpi::communicator world_, std::vector<double> input_array_A_);
void SeqHoaraSort(std::vector<double>::iterator first, std::vector<double>::iterator last);
void HoaraSort(std::vector<double>::iterator first, std::vector<double>::iterator last);
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
  boost::mpi::communicator world_;
};
}  // namespace deryabin_m_hoare_sort_simple_merge_mpi
