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
  std::vector<double> input_;
  std::vector<double> output_;
  static const size_t kParallelThreshold = 1000;
  size_t chunk_count_;

  void QuickSort(std::vector<double>& arr, size_t low, size_t high);
  void ParallelQuickSort(std::vector<double>& arr, size_t low, size_t high);
  static size_t Partition(std::vector<double>& arr, size_t low, size_t high);
};

}  // namespace shuravina_o_hoare_simple_merger_tbb