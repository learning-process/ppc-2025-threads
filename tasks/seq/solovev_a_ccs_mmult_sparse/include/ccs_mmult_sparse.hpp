#pragma once

#include <complex>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_matrix {
struct MatrixInCCS_Sparse {
  std::vector<std::complex<double>> val{};
  std::vector<int> row;
  std::vector<int> col_p;

  int r_n;
  int c_n;
  int n_z;

  MatrixInCCS_Sparse(int _r_n = 0, int _c_n = 0, int _n_z = 0) {
    c_n = _c_n;
    r_n = _r_n;
    n_z = _n_z;
    row.resize(n_z);
    col_p.resize(r_n + 1);
    val.resize(n_z);
  }
};

class Seq_MatMultCCS : public ppc::core::Task {
 public:
  explicit Seq_MatMultCCS(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  MatrixInCCS_Sparse *M1, *M2, *M3;
};
}  // namespace solovev_a_matrix