#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_all {
static uint32_t ConvertIntToUint(int num);
static int ConvertUintToInt(uint32_t unum);
void RadixSort(std::vector<uint32_t>& uns_vec);

class RadixBatcherSortTaskAll : public ppc::core::Task {
 public:
  explicit RadixBatcherSortTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<uint32_t> unsigned_data_;
  int size_;
  boost::mpi::communicator world_;
  size_t global_original_size_;
};
}  // namespace opolin_d_radix_batcher_sort_all