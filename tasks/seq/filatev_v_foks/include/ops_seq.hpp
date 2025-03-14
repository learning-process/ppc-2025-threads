#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace filatev_v_foks_seq {

struct matrix_size {
  size_t n;
  size_t m;
};

class Focks : public ppc::core::Task {
 public:
  explicit Focks(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  matrix_size size_a_{};
  matrix_size size_b_{};
  matrix_size size_c_{};

  size_t size_block_{};
  size_t size_{};

  std::vector<double> matrix_a_;
  std::vector<double> matrix_b_;
  std::vector<double> matrix_c_;

};

}  // namespace filatev_v_foks_seq