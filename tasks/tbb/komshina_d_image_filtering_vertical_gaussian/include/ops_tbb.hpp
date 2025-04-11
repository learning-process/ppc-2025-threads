#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_image_filtering_vertical_gaussian_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ApplyFilter(int y, int x);

  std::vector<unsigned char> input_;
  std::vector<unsigned char> output_;
  std::size_t height_;
  std::size_t width_;

  std::vector<float> kernel_;
};

}  // namespace komshina_d_image_filtering_vertical_gaussian_tbb