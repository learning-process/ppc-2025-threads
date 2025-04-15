#ifndef OPS_TBB_HPP
#define OPS_TBB_HPP

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Bigint = long long;
using namespace std;

namespace belov_a_radix_batcher_mergesort_tbb {

class RadixBatcherMergesortParallel : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void Sort(std::span<Bigint> arr);

 private:
  std::vector<Bigint> array_;  // input unsorted numbers array
  size_t n_ = 0;               // array size

  static void RadixSort(std::vector<Bigint>& arr, bool invert);
  static void CountingSort(std::vector<Bigint>& arr, Bigint digit_place);
  static int GetNumberDigitCapacity(Bigint num);

  static void SortParallel(std::vector<Bigint>& arr);
  static void BatcherMergeParallel(std::vector<Bigint>& arr);
};

}  // namespace belov_a_radix_batcher_mergesort_tbb

#endif  // OPS_TBB_HPP