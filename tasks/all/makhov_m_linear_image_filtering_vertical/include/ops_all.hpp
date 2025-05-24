#pragma once

#include <omp.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_linear_image_filtering_vertical_all {

void BlurColumn(const uint8_t* src, uint8_t* dst, int width, int height, int x);

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
  std::vector<uint8_t> input_image;
  std::vector<uint8_t> output_image;
  std::uint32_t input_size;
  std::uint32_t height;
  std::uint32_t width;
  int rank_;
  int size_;
};

}  // namespace makhov_m_linear_image_filtering_vertical_all