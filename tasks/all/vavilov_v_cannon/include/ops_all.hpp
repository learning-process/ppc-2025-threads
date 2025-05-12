#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_cannon_all {
class CannonALL : public ppc::core::Task {
 public:
  explicit CannonALL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int N_;
  int block_size_;
  int num_blocks_;
  //int num_rows_;
  //int num_cols_;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  boost::mpi::communicator world_;

  void InitialShift(std::vector<double>& local_A, std::vector<double>& local_B);
  void BlockMultiply(const std::vector<double>& local_A, const std::vector<double>& local_B,
                     std::vector<double>& local_C);
  void ShiftBlocksone();
  void InitialShiftone();
  void BlockMultiplyone();
  void ShiftBlocks(std::vector<double>& local_A, std::vector<double>& local_B);
  static int find_compatible_q(int size, int N);
  static void extract_block(const std::vector<double>& matrix, double* block, int N, int K, int block_row,
                            int block_col);
};
}  // namespace vavilov_v_cannon_all
