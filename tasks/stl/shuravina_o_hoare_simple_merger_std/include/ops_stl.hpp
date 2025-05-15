#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(std::shared_ptr<ppc::core::TaskData> task_data);

  bool Validation() override;
  bool PreProcessing() override;
  bool Run() override;
  bool PostProcessing() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;
  size_t chunk_count_;
  size_t min_chunk_size_;

  static void QuickSort(std::vector<double>& arr, size_t left, size_t right);
  static void MergeHelper(std::vector<double>& arr, size_t left, size_t mid, size_t right);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shuravina_o_hoare_simple_merger_stl