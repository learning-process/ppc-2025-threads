#pragma once

#include <oneapi/tbb/task_group.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_hoare_sort_simple_merge_mpi {

void HoaraSort(std::vector<double>& a, size_t first, size_t last);
void HoaraSort(std::vector<double>& a, size_t first, size_t last, tbb::task_group& tg, size_t available_threads);
void MergeTwoParts(std::vector<double>& a, size_t first, size_t last, tbb::task_group& tg, size_t available_threads);

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
  size_t chunk_count_;
  boost::mpi::communicator world;
};
}  // namespace deryabin_m_hoare_sort_simple_merge_mpi
