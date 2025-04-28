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
  int row = rank / grid_size;
  int col = rank % grid_size;
  int a_dest = (row * grid_size + (col - row + grid_size) % grid_size);
  int a_src = (row * grid_size + (col + row) % grid_size);
  int b_dest = (((row - col + grid_size) % grid_size) * grid_size + col);
  int b_src = (((row + col) % grid_size) * grid_size + col);
  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  if (a_dest != rank) {
    world_.send(a_dest, 0, local_A.data(), block_size_ * block_size_);
    world_.recv(a_src, 0, tmp_A.data(), block_size_ * block_size_);
    local_A = tmp_A;
  }
  if (b_dest != rank) {
    world_.send(b_dest, 1, local_B.data(), block_size_ * block_size_);
    world_.recv(b_src, 1, tmp_B.data(), block_size_ * block_size_);
    local_B = tmp_B;
  }
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
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  int left_dest = (col == 0) ? (row * grid_size + grid_size - 1) : (rank - 1);
  int left_src = (col == grid_size - 1) ? (row * grid_size) : (rank + 1);
  int up_dest = (row == 0) ? ((grid_size - 1) * grid_size + col) : (rank - grid_size);
  int up_src = (row == grid_size - 1) ? col : (rank + grid_size);

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  if (left_dest != rank) {
    world_.send(left_dest, 2, local_A.data(), block_size_ * block_size_);
    world_.recv(left_src, 2, tmp_A.data(), block_size_ * block_size_);
    local_A = tmp_A;
  }
  if (up_dest != rank) {
    world_.send(up_dest, 3, local_B.data(), block_size_ * block_size_);
    world_.recv(up_src, 3, tmp_B.data(), block_size_ * block_size_);
    local_B = tmp_B;
  }
}

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  int grid_size = static_cast<int>(std::sqrt(size));
  if (grid_size * grid_size != size) {
    if (rank == 0) {
      std::cerr << "Number of processes must be a perfect square" << std::endl;
    }
    return false;
  }
  if (N_ % grid_size != 0) {
    if (rank == 0) {
      std::cerr << "Matrix size must be divisible by grid size" << std::endl;
    }
    return false;
  }

  num_blocks_ = grid_size;
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;
  boost::mpi::broadcast(world_, A_, 0);
  boost::mpi::broadcast(world_, B_, 0);

  std::vector<double> local_A(block_size_sq, 0);
  std::vector<double> local_B(block_size_sq, 0);
  std::vector<double> local_C(block_size_sq, 0);

  int row = rank / grid_size;
  int col = rank % grid_size;
  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      int global_i = row * block_size_ + i;
      int global_j = col * block_size_ + j;
      if (global_i < N_ && global_j < N_) {
        local_A[i * block_size_ + j] = A_[global_i * N_ + global_j];
        local_B[i * block_size_ + j] = B_[global_i * N_ + global_j];
      }
    }
  }
  InitialShift(local_A, local_B);
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply(local_A, local_B, local_C);
    if (iter < num_blocks_ - 1) {
      ShiftBlocks(local_A, local_B);
    }
  }

  std::vector<double> gathered_C(size * block_size_sq);
  boost::mpi::all_gather(world_, local_C.data(), block_size_sq, gathered_C);
  std::vector<std::vector<double>> all_C(size, std::vector<double>(block_size_sq));
  for (int p = 0; p < size; ++p) {
    std::copy(gathered_C.begin() + p * block_size_sq, gathered_C.begin() + (p + 1) * block_size_sq, all_C[p].begin());
  }

  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      int row_p = p / grid_size;
      int col_p = p % grid_size;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          int global_i = row_p * block_size_ + i;
          int global_j = col_p * block_size_ + j;
          if (global_i < N_ && global_j < N_) {
            C_[global_i * N_ + global_j] = all_C[p][i * block_size_ + j];
          }
        }
      }
    }
  }
  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
