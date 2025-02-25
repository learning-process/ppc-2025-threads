#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_hoare_sort {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  unsigned int *input_{};
  unsigned int n_{};
  std::vector<int> output_;
  unsigned int value_;
  void HoareSort(unsigned int *s_vec, int first, int last);
  int Partition(unsigned int *s_vec, int first, int last);
};

}  // namespace vershinina_a_hoare_sort