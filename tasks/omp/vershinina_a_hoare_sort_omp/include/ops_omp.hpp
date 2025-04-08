#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_hoare_sort_omp {
class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  int n_{};
  int value_;
  std::vector<double> res_;
  void HoareSort(double *s_vec, int first, int last);
  int Partition(double *s_vec, int first, int last);
};

}  // namespace vershinina_a_hoare_sort_omp