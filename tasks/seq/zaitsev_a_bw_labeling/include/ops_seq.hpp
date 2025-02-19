#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zaitsev_a_labeling {

class Labeler : public ppc::core::Task {
 public:
  explicit Labeler(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> image_;
  std::vector<int> labels_;
  std::vector<int> equivalences_;
  std::vector<int> replacements_;
  unsigned int width_;
  unsigned int height_;
  unsigned int size_;
  int current_label_;

  void ComputeLabel(unsigned int i);
  void LabelingRasterScan();
  void PrepareReplacements();
  void PerformReplacements();
};

}  // namespace zaitsev_a_labeling