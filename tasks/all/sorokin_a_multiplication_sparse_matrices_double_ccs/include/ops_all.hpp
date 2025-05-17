#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int M_;
  int K_;
  int N_;
  std::vector<double> A_values_;
  std::vector<int> A_row_indices_;
  std::vector<int> A_col_ptr_;
  std::vector<double> B_values_;
  std::vector<int> B_row_indices_;
  std::vector<int> B_col_ptr_;
  std::vector<double> C_values_;
  std::vector<int> C_row_indices_;
  std::vector<int> C_col_ptr_;
  boost::mpi::communicator world_;
};

void MultiplyCCS(boost::mpi::communicator &world, const std::vector<double> &a_values,
                 const std::vector<int> &a_row_indices, int m, const std::vector<int> &a_col_ptr,
                 const std::vector<double> &b_values, const std::vector<int> &b_row_indices, int k,
                 const std::vector<int> &b_col_ptr, std::vector<double> &c_values, std::vector<int> &c_row_indices,
                 int n, std::vector<int> &c_col_ptr);
void DistributeBColumns(boost::mpi::communicator &world, int rank, int size, const std::vector<double> &b_values,
                        const std::vector<int> &b_row_indices, const std::vector<int> &b_col_ptr, int n,
                        int base_cols_per_proc, int remainder, std::vector<double> &local_b_values,
                        std::vector<int> &local_b_row_indices, std::vector<int> &local_b_col_ptr);
void CalculateLocalNNZ(const std::vector<int> &a_row_indices, const std::vector<int> &a_col_ptr,
                       const std::vector<int> &local_b_row_indices, const std::vector<int> &local_b_col_ptr, int m,
                       int num_local_cols, std::vector<int> &local_nnz);
void ComputeLocalC(const std::vector<double> &a_values, const std::vector<int> &a_row_indices,
                   const std::vector<int> &a_col_ptr, const std::vector<double> &local_b_values,
                   const std::vector<int> &local_b_row_indices, const std::vector<int> &local_b_col_ptr, int m,
                   int num_local_cols, const std::vector<int> &local_nnz, std::vector<double> &local_c_values,
                   std::vector<int> &local_c_row_indices, std::vector<int> &local_c_col_ptr);
void GatherResults(boost::mpi::communicator &world, int rank, int size, int n, int base_cols_per_proc, int remainder,
                   int num_local_cols, int start_col, const std::vector<int> &local_nnz,
                   const std::vector<double> &local_c_values, const std::vector<int> &local_c_row_indices,
                   const std::vector<int> &local_c_col_ptr, std::vector<double> &c_values,
                   std::vector<int> &c_row_indices, std::vector<int> &c_col_ptr);
void SendLocalData(boost::mpi::communicator &world, int num_local_cols, int start_col,
                   const std::vector<int> &local_nnz, const std::vector<int> &local_c_col_ptr,
                   const std::vector<int> &local_c_row_indices, const std::vector<double> &local_c_values);
void GatherData(boost::mpi::communicator &world, int size, int base_cols_per_proc, int remainder,
                const std::vector<int> &local_c_col_ptr, const std::vector<double> &local_c_values,
                const std::vector<int> &local_c_row_indices, std::vector<double> &c_values,
                std::vector<int> &c_row_indices, const std::vector<int> &c_col_ptr, const std::vector<int> &gather_nnz);
void BuildColPtr(const std::vector<int> &gather_nnz, std::vector<int> &c_col_ptr);
void GatherNnz(boost::mpi::communicator &world, int size, int base_cols_per_proc, int remainder,
               const std::vector<int> &local_nnz, std::vector<int> &gather_nnz);
}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_all
