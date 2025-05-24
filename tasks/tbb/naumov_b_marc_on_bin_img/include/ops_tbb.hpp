#pragma once

#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

namespace naumov_b_marc_on_bin_img_tbb {

std::vector<int> GenerateRandomBinaryMatrix(int rows, int cols, double probability = 0.5);
std::vector<int> GenerateSparseBinaryMatrix(int rows, int cols, double probability = 0.1);
std::vector<int> GenerateDenseBinaryMatrix(int rows, int cols, double probability = 0.9);

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ProcessPixel(int row, int col);
  void AssignNewLabel(int row, int col);
  void AssignMinLabel(int row, int col, const std::vector<int>& neighbors);
  std::vector<int> FindAdjacentLabels(int row, int col);
  int FindRoot(int label);
  void UnionLabels(int label1, int label2);

  int rows_{};
  int cols_{};
  std::vector<int> input_image_;
  std::vector<int> output_image_;
  std::vector<int> label_parent_;
  std::atomic_int current_label_;
  std::mutex label_mutex_;
};

}  // namespace naumov_b_marc_on_bin_img_tbb