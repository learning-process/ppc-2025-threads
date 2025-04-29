#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;

  void QuickSort(std::vector<int>& arr, int left, int right);
  void Merge(std::vector<int>& arr, int left, int mid, int right);
  void ParallelQuickSort(std::vector<int>& arr, int left, int right);
};

}  // namespace shuravina_o_hoare_simple_merger_stl