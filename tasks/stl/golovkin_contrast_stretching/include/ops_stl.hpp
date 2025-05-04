#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace golovkin_contrast_stretching {

class ContrastStretchingSTL : public ppc::core::Task {
 public:
  explicit ContrastStretchingSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_image_;
  std::vector<uint8_t> output_image_;
  size_t image_size_ = 0;

  uint8_t min_val_ = 0;
  uint8_t max_val_ = 0;
};

}  // namespace golovkin_contrast_stretching