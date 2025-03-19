#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_marc_on_bin_image_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ProcessBlock(int start_row, int start_col, int block_rows, int block_cols);
  void MergeLabels();

  std::vector<int> FindAdjacentLabels(int row, int col);
  void AssignLabel(int row, int col, int& current_label);
  int FindRoot(int label);
  void UnionLabels(int label1, int label2);
  void CalculateBlockSize();

  int rows_{};
  int cols_{};
  std::vector<int> input_image_;
  std::vector<int> output_image_;
  std::vector<int> label_parent_;
  int block_size_ = 64;
  int current_label_ = 0;
};

}  // namespace naumov_b_marc_on_bin_image_seq