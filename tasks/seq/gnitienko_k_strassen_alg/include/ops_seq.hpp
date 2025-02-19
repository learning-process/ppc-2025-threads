#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gnitienko_k_strassen_algorithm {

class StrassenAlgSeq : public ppc::core::Task {
 public:
  explicit StrassenAlgSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_1, input_2, output_;
  int size_{};
  int TRIVIAL_MULTIPLICATION_BOUND = 8;
  int extend = 0;

  void TrivialMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size);
  void StrassenMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size);
  void AddMatrix(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size);
  void SubMatrix(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size);
};

}  // namespace gnitienko_k_strassen_algorithm