#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_sobel_filter_all {

struct RGB {
  int R{};
  int G{};
  int B{};
};

std::vector<int> ToGrayScaleImg(std::vector<RGB>& color_img, size_t width, size_t height);

int GetPixelSafe(const std::vector<int>& img, size_t x, size_t y, size_t width, size_t height);

class SobelFilterALL : public ppc::core::Task {
 public:
  explicit SobelFilterALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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
  boost::mpi::communicator world_;

  std::vector<int> local_image_;
};

}  // namespace frolova_e_sobel_filter_all
