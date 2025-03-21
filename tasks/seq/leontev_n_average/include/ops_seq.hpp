#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_average_seq {
template <class InOutType>
class VecAvgSequential : public ppc::core::Task {
 public:
  explicit VecAvgSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<InOutType> input_;
  InOutType res_{};
};

}  // namespace leontev_n_average_seq
