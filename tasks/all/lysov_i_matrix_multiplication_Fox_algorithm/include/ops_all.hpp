#pragma once

#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb {
void extract_submatrix_block(const std::vector<double>& matrix, double* block, int total_columns, int block_size,
                             int block_row_index, int block_col_index);

void multiply_matrix_blocks(const double* A, const double* B, double* C, int block_size);
void perform_fox_algorithm_step(boost::mpi::communicator& my_world, int rank, int cnt_work_process, int K,
                                std::vector<double>& local_A, std::vector<double>& local_B,
                                std::vector<double>& local_C);
void TrivialMatrixMultiplication(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b,
                                 std::vector<double>& result_matrix, size_t matrix_size);
std::vector<double> GetRandomMatrix(size_t size, int min_gen_value, int max_gen_value);
class TestTaskMPITBB : public ppc::core::Task {
 public:
  explicit TestTaskMPITBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> a_, b_;
  std::vector<double> resultC_;

  std::size_t n_ = 0;
  std::size_t block_size_ = 0;
  int elements{};

  std::vector<double> resultC;
  boost::mpi::communicator world_;
};

}  // namespace lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb
