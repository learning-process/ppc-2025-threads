#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace odintsov_m_mulmatix_cannon_seq {

class MulMatrixCannonSequential : public ppc::core::Task {
 public:
  explicit MulMatrixCannonSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void shiftRow(std::vector<double>& matrix, int root, int row, int shift);
  void shiftColumn(std::vector<double>& matrix, int root, int col, int shift);
  void shiftBlocksUp(std::vector<double>& matrix, int root, int block_sz);
  void shiftBlocksLeft(std::vector<double>& matrix, int root, int block_sz);
  bool is_squere(int num);
  int get_block_size(int N);
  std::vector<double> matrixA, matrixB;
  int szA = 0, szB = 0;
  int block_sz = 0;
  std::vector<double> matrixC;
};

}  // namespace odintsov_m_mulmatix_cannon_seq
