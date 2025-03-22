#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tsatsyn_a_radix_sort_simple_merge_all {
std::vector<double> GetRandomVector(int sz, int a, int b);
class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
  std::vector<double> local_data_;
  std::vector<double> output_;
  int rc_size_{};
  boost::mpi::communicator world_;
};

}  // namespace tsatsyn_a_radix_sort_simple_merge_all