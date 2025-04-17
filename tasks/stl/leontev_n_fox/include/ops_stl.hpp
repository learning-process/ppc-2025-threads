#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_fox_stl {

std::vector<double> mat_mul(std::vector<double>& a, std::vector<double>& b, size_t n);

class FoxSTL : public ppc::core::Task {
 public:
  explicit FoxSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double at_a(size_t i, size_t j) const;
  double at_b(size_t i, size_t j) const;
  void mat_mul_blocks(size_t a_posX, size_t a_posY, size_t b_posX, size_t b_posY, size_t c_posX, size_t c_posY,
                      size_t size);
  std::vector<double> input_a_;
  std::vector<double> input_b_;
  std::vector<double> output_;
  size_t n;
};

}  // namespace leontev_n_fox_stl
