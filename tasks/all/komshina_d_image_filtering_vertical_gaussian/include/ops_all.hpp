#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_image_filtering_vertical_gaussian_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<unsigned char> FilterLocalRegion(std::size_t start_row, std::size_t end_row) const;

  boost::mpi::communicator world_;
  std::vector<unsigned char> input_;
  std::vector<unsigned char> output_;
  std::vector<float> kernel_;
  std::size_t height_{};
  std::size_t width_{};
};

}  // namespace komshina_d_image_filtering_vertical_gaussian_all
