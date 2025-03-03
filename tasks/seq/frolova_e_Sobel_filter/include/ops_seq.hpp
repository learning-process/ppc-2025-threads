#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_Sobel_filter_seq {

struct RGB {

  int R{};
  int G{};
  int B{};

};

std::vector<int> toGrayScaleImg(std::vector<RGB>& colorImg, size_t width, size_t height);
int Clamp(int value, int minVal, int maxVal);
std::vector<int> genRGBpicture(size_t width, size_t height, size_t seed = 0);

class SobelFilterSequential : public ppc::core::Task {
 public:
  explicit SobelFilterSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:

  std::vector<RGB> picture{};
  std::vector<int> grayscaleImage;
  size_t width{};
  size_t height{};
  std::vector<int> resImage;

};	

}  // namespace frolova_e_Sobel_filter_seq