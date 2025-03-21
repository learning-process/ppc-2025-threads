#pragma once

#include <utility>
#include <cstdint>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_image_filtering_vertical_gaussian_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  uint32_t height_;
  uint32_t width_;
  
  std::vector<float> kernel_;
};

}  // namespace komshina_d_image_filtering_vertical_gaussian_seq