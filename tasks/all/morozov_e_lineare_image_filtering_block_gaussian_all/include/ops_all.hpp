#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace morozov_e_lineare_image_filtering_block_gaussian_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), n_{0}, m_{0} {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, res_;
  int n_{0}, m_{0};
  boost::mpi::communicator world_;
};

}  // namespace morozov_e_lineare_image_filtering_block_gaussian_all