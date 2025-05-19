#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  static const size_t kParallelThreshold = 1000;

  void QuickSort(std::vector<int>& arr, int low, int high);
  void ParallelQuickSort(std::vector<int>& arr, int low, int high);
  static int Partition(std::vector<int>& arr, int low, int high);
};

}  // namespace shuravina_o_hoare_simple_merger_tbb