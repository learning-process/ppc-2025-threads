#pragma once

#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_linear_stretching {

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  cv::Mat inputImage_, outputImage_;

  int rc_size_{};
};

}  // namespace tarakanov_d_linear_stretching
