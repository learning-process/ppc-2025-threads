#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Bigint = long long;
using namespace std;

namespace belov_a_radix_batcher_mergesort_seq {

class RadixBatcherMergesortSequential : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortSequential(std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void Sort(vector<Bigint>& arr);

  std::vector<Bigint> GenerateMixedValuesArray(int n);
  vector<Bigint> GenerateIntArray(int n);
  vector<Bigint> GenerateBigintArray(int n);

 private:
  vector<Bigint> array_;  // input unsorted numbers array
  size_t n_ = 0;          // array size

  void static RadixSort(vector<Bigint>& arr, bool invert);
  static void CountingSort(vector<Bigint>& arr, Bigint digit_place);
  static int GetNumberDigitCapacity(Bigint num);
};

}  // namespace belov_a_radix_batcher_mergesort_seq

#endif  // OPS_SEQ_HPP