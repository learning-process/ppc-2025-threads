#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gnitienko_k_strassen_algorithm_omp {

class StrassenAlgOpenMP : public ppc::core::Task {
 public:
  explicit StrassenAlgOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_1_;
  std::vector<double> input_2_;
  std::vector<double> output_;
  int size_{};
  int TRIVIAL_MULTIPLICATION_BOUND_ = 32;
  int extend_ = 0;

  // NOLINTBEGIN(readability-convert-member-functions-to-static)
  void TrivialMultiply(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int size);
  // NOLINTEND(readability-convert-member-functions-to-static)
  void StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int size);
  static void AddMatrix(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int size);
  static void SubMatrix(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int size);
};

}  // namespace gnitienko_k_strassen_algorithm_omp