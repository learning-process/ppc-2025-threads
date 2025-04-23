#pragma once

#include <utility>
#include <vector>

#include "all/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"
#include "core/task/include/task.hpp"
namespace sarafanov_m_canon_mat_mul_all {
class CanonMatMulALL : public ppc::core::Task {
  CanonMatrix a_matrix_;
  CanonMatrix b_matrix_;
  CanonMatrix c_matrix_;
  static constexpr double kInaccuracy = 0.001;

 public:
  explicit CanonMatMulALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool CheckSquareSize(int number);
  static std::vector<double> ConvertToSquareMatrix(int need_size, MatrixType type, const std::vector<double>& matrx);
};
}  // namespace sarafanov_m_canon_mat_mul_all