#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_image_filter_horiz_gaussian3x3_all {

class GaussianFilterALL : public ppc::core::Task {
 public:
  explicit GaussianFilterALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;
  int width_;
  int height_;
  std::vector<int> kernel_;
  boost::mpi::communicator world_;
};

}  // namespace titov_s_image_filter_horiz_gaussian3x3_all