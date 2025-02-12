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
  std::vector<std::vector<int>> image_;   // original image
  std::vector<std::vector<int>> labels_;  // label map
  std::map<int, int> equivalencies_;      // equvivalency map
  unsigned int width_;                    // image width
  unsigned int height_;                   // image height

  void ComputeLabel(unsigned int i, unsigned int j, int& current_label);
  void RasterScan();
  void EquivReplace();
};

}  // namespace zaitsev_a_labeling