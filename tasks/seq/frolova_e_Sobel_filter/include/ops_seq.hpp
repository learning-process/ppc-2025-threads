#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_sobel_filter_seq {

struct RGB {
  int R{};
  int G{};
  int B{};
};

std::vector<int> ToGrayScaleImg(std::vector<RGB> &colorImg, size_t width, size_t height);
int Clamp(int value, int min_val, int max_val);
std::vector<int> GenRgbPicture(size_t width, size_t height, size_t seed = 0);

class SobelFilterSequential : public ppc::core::Task {
 public:
  explicit SobelFilterSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<RGB> picture_;
  std::vector<int> grayscale_image_;
  size_t width_{};
  size_t height_{};
  std::vector<int> res_image_;
};

}  // namespace frolova_e_sobel_filter_seq