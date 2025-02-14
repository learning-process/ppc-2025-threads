#pragma once

#include <map>
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
  std::vector<std::vector<int>> image_;
  std::vector<std::vector<int>> labels_;
  std::map<int, int> equivalencies_;
  unsigned int width_;
  unsigned int height_;

  void ComputeLabel(unsigned int i, unsigned int j, int& current_label);
  void LabelingRasterScan();
  void EquivReplaceRasterScan();
};

}  // namespace zaitsev_a_labeling