#pragma once

#include <utility>
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
  uint16_t height_;
  uint16_t width_;
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  std::vector<float> kernel_;
};

}  // namespace komshina_d_image_filtering_vertical_gaussian_seq