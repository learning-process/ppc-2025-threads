#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_tbb {

class HoareSortTBB : public ppc::core::Task {
 public:
  explicit HoareSortTBB(std::shared_ptr<ppc::core::TaskData> task_data);
  bool Validation() override;
  bool PreProcessing() override;
  bool Run() override;
  bool PostProcessing() override;

 private:
  std::vector<int> data_;
  static constexpr size_t kThreshold = 10000;

  void SequentialQuickSort(int* arr, size_t left, size_t right);
  size_t Partition(int* arr, size_t left, size_t right);
  void ParallelQuickSort(int* arr, size_t left, size_t right);
};

}  // namespace shuravina_o_hoare_simple_merger_tbb