#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_histogram_linear_stretching_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int min_val_ = 0;
  int max_val_ = 255;
  std::vector<uint8_t> input_image_, result_image_;
  boost::mpi::communicator world_;

  void scatter_data(std::vector<uint8_t>& local_data);
  void find_min_max(const std::vector<uint8_t>& local_data, int& min_val, int& max_val);
  void stretch_histogram(std::vector<uint8_t>& local_data, int min_val, int max_val);
  void gather_results(const std::vector<uint8_t>& local_data);
};

}  // namespace varfolomeev_g_histogram_linear_stretching_all