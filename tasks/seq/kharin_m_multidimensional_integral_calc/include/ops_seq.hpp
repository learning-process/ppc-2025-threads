#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kharin_m_multidimensional_integral_calc_seq {

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int output_result_{0};  // Здесь сохраняется результат вычисления интеграла.
  size_t grid_size_{0};   // Размер стороны квадратной сетки (n x n).
};

}  // namespace kharin_m_multidimensional_integral_calc_seq