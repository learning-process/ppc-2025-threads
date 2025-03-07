#pragma once

#include <map>
#include <set>
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
  std::vector<uint8_t> image_;
  std::vector<uint16_t> labels_;
  unsigned int width_;
  unsigned int height_;
  unsigned int size_;

  void GetNeighbours(unsigned int i, std::vector<uint16_t>& neighbours);
  void ComputeLabel(unsigned int i, std::map<uint16_t, std::set<uint16_t>>& eqs, uint16_t& current_label);
  void LabelingRasterScan(std::map<uint16_t, std::set<uint16_t>>& eqs, uint16_t& current_label);
  static void CalculateReplacements(std::vector<uint16_t>& replacements, std::map<uint16_t, std::set<uint16_t>>& eqs,
                                    uint16_t& current_label);
  void PerformReplacements(std::vector<uint16_t>& replacements);
};

}  // namespace zaitsev_a_labeling