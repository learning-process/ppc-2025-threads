#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp.h"

namespace titov_s_image_filter_horiz_gaussian3x3_omp {

class ImageFilterOMP : public ppc::core::Task {
 public:
  explicit ImageFilterOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;
  int width_;
  int height_;
  const size_t kernel_size_ = 3;
  std::vector<int> kernel_;
};

}  // namespace titov_s_image_filter_horiz_gaussian3x3_omp