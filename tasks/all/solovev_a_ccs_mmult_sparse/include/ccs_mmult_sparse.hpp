#pragma once

#include <boost/serialization/access.hpp>
#include <complex>
#include <memory>
#include <utility>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace solovev_a_matrix_all {

struct MatrixInCcsSparse {
  int r_n;
  int c_n;
  int n_z;

  std::vector<std::complex<double>> val;
  std::vector<int> row;
  std::vector<int> col_p;

  MatrixInCcsSparse(int r_nn = 0, int c_nn = 0, int n_zz = 0)
      : r_n(r_nn), c_n(c_nn), n_z(n_zz), val(n_zz), row(n_zz), col_p(c_n + 1) {}

  friend class boost::serialization::access;

  // clang-format off
  // NOLINTBEGIN(*)
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar & r_n;
    ar & c_n;
    ar & n_z;
    ar & val;
    ar & row;
    ar & col_p;
  }
  // NOLINTEND(*)
  // clang-format on
};

class SeqMatMultCcs : public ppc::core::Task {
 public:
  explicit SeqMatMultCcs(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ProcessPhase1(SeqMatMultCcs* self, int col, std::vector<int>& available);
  static void ProcessPhase2(SeqMatMultCcs* self, int col, std::vector<int>& available,
                            std::vector<std::complex<double>>& cask);
  static void NotifyCompletion(SeqMatMultCcs* self);
  static void WorkerLoop(SeqMatMultCcs* self);

 private:
  MatrixInCcsSparse *M1_, *M2_;
  MatrixInCcsSparse M3_;

  boost::mpi::communicator world_;
};

}  // namespace solovev_a_matrix_all
