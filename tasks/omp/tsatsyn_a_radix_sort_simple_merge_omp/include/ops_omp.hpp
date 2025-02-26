#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tsatsyn_a_radix_sort_simple_merge_omp {
std::vector<double> GetRandomVector(int sz, int a, int b);
class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
  std::vector<double> output_;
};

}  // namespace tsatsyn_a_radix_sort_simple_merge_omp