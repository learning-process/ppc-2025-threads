#include "all/vavilov_v_cannon/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

namespace mpi = boost::mpi;

bool vavilov_v_cannon_all::CannonALL::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(std::sqrt(N_));
  block_size_ = N_ / num_blocks_;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_all::CannonALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;

  int grid_rows = num_blocks_row_;
  int grid_cols = num_blocks_col_;

  int row = rank / grid_cols;
  int col = rank % grid_cols;

  int a_dest = (row * grid_cols + (col - row + grid_cols) % grid_cols);
  int a_src = (row * grid_cols + (col + row) % grid_cols);
  int b_dest = (((row - col + grid_rows) % grid_rows) * grid_cols + col);
  int b_src = (((row + col) % grid_rows) * grid_cols + col);

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);
  if (a_dest != rank) {
    world_.send(a_dest, 0, local_A.data(), block_size_ * block_size_);
    world_.recv(a_src, 0, tmp_A.data(), block_size_ * block_size_);
    local_A = tmp_A;
  }
  world_.barrier();
  if (b_dest != rank) {
    world_.send(b_dest, 1, local_B.data(), block_size_ * block_size_);
    world_.recv(b_src, 1, tmp_B.data(), block_size_ * block_size_);
    local_B = tmp_B;
  }
  world_.barrier();
}

void vavilov_v_cannon_all::CannonALL::BlockMultiply(const std::vector<double>& local_A,
                                                    const std::vector<double>& local_B, std::vector<double>& local_C) {
#pragma omp parallel for
  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      double temp = 0.0;
      for (int k = 0; k < block_size_; ++k) {
        temp += local_A[i * block_size_ + k] * local_B[k * block_size_ + j];
      }
      local_C[i * block_size_ + j] += temp;
    }
  }
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_rows = num_blocks_row_;
  int grid_cols = num_blocks_col_;
  int row = rank / grid_cols;
  int col = rank % grid_cols;

  int left_dest = (col == 0) ? (row * grid_cols + grid_cols - 1) : (rank - 1);
  int left_src = (col == grid_cols - 1) ? (row * grid_cols) : (rank + 1);
  int up_dest = (row == 0) ? ((grid_rows - 1) * grid_cols + col) : (rank - grid_cols);
  int up_src = (row == grid_rows - 1) ? col : (rank + grid_cols);

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  if (left_dest != rank) {
    world_.send(left_dest, 2, local_A.data(), block_size_ * block_size_);
    world_.recv(left_src, 2, tmp_A.data(), block_size_ * block_size_);
    local_A = tmp_A;
  }
  world_.barrier();
  if (up_dest != rank) {
    world_.send(up_dest, 3, local_B.data(), block_size_ * block_size_);
    world_.recv(up_src, 3, tmp_B.data(), block_size_ * block_size_);
    local_B = tmp_B;
  }
  world_.barrier();
}

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  int num_blocks_row_ = static_cast<int>(std::sqrt(size));
  while (num_blocks_row_ > 0 && size % num_blocks_row_ != 0) {
    num_blocks_row_--;
  }
  if (num_blocks_row_ == 0) {
    if (rank == 0) {
      std::cerr << "Cannot form a valid grid for " << size << " processes" << std::endl;
    }
    world_.barrier();
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
  int num_blocks_col_ = size / num_blocks_row_;
  if (N_ % num_blocks_row_ != 0 || N_ % num_blocks_col_ != 0) {
    world_.barrier();
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
  num_blocks_ = std::max(num_blocks_row_, num_blocks_col_);
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;
  boost::mpi::broadcast(world_, A_.data(), N_ * N_, 0);
  boost::mpi::broadcast(world_, B_.data(), N_ * N_, 0);

  std::vector<double> local_A(block_size_sq, 0);
  std::vector<double> local_B(block_size_sq, 0);
  std::vector<double> local_C(block_size_sq, 0);

  int row = rank / num_blocks_col_;
  int col = rank % num_blocks_col_;
  int block_idx = (row * num_blocks_ + col) * block_size_sq;
  std::copy(A_.begin() + block_idx, A_.begin() + block_idx + block_size_sq, local_A.begin());
  std::copy(B_.begin() + block_idx, B_.begin() + block_idx + block_size_sq, local_B.begin());

  InitialShift(local_A, local_B);
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply(local_A, local_B, local_C);
    if (iter < num_blocks_ - 1) {
      ShiftBlocks(local_A, local_B);
    }
  }

  std::vector<std::vector<double>> all_C;
  if (rank == 0) {
    all_C.resize(size, std::vector<double>(block_size_sq));
  }
  boost::mpi::gather(world_, local_C, all_C, 0);

  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      int row_p = p / num_blocks_col_;
      int col_p = p % num_blocks_col_;
      int block_idx_p = (row_p * num_blocks_ + col_p) * block_size_sq;
      std::copy(all_C[p].begin(), all_C[p].end(), C_.begin() + block_idx_p);
    }
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
