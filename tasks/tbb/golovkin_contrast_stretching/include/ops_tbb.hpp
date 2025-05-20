// Golovkin Maksim
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/task/include/task.hpp"

namespace golovkin_contrast_stretching {

template <typename PixelType = uint8_t>
class ContrastStretchingTBB : public ppc::core::Task {
 public:
  explicit ContrastStretchingTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<PixelType> input_image_;
  std::vector<PixelType> output_image_;
  size_t image_size_ = 0;

  PixelType min_val_ = 0;
  PixelType max_val_ = 0;
};

}  // namespace golovkin_contrast_stretching