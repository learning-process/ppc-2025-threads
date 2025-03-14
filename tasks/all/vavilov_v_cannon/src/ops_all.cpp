#include "all/vavilov_v_cannon/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <vector>

namespace mpi = boost::mpi;

bool vavilov_v_cannon_all::CannonALL::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(std::sqrt(N_));
  block_size_ = N_ / num_blocks_;

  auto *a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *b = reinterpret_cast<double *>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_all::CannonALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

void vavilov_v_cannon_all::CannonALL::InitialShift() {
  int row = rank_ / grid_size_;
  int col = rank_ % grid_size_;

  int a_dest = ((row + col) % grid_size_) + row * grid_size_;
  int a_source = ((row - col + grid_size_) % grid_size_) + row * grid_size_;

  int b_dest = col + ((row + col) % grid_size_) * grid_size_;
  int b_source = col + ((row - col + grid_size_) % grid_size_) * grid_size_;

  std::vector<double> tmp_a = local_A_;
  std::vector<double> tmp_b = local_B_;

  world.send(a_dest, 0, tmp_a);
  world.send(b_dest, 1, tmp_b);
  world.recv(a_source, 0, local_A_);
  world.recv(b_source, 1, local_B_);
}

void vavilov_v_cannon_all::CannonALL::BlockMultiply() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      double temp = 0.0;
      for (int k = 0; k < block_size_; ++k) {
        temp += local_A_[i * block_size_ + k] * local_B_[k * block_size_ + j];
      }
      local_C_[i * block_size_ + j] += temp;
    }
  }
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks() {
  int row = rank_ / grid_size_;
  int col = rank_ % grid_size_;

  int left = (col == 0) ? row * grid_size_ + grid_size_ - 1 : rank_ - 1;
  int right = (col == grid_size_ - 1) ? row * grid_size_ : rank_ + 1;

  int up = (row == 0) ? (grid_size_ - 1) * grid_size_ + col : rank_ - grid_size_;
  int down = (row == grid_size_ - 1) ? col : rank_ + grid_size_;

  std::vector<double> tmp_a = local_A_;
  std::vector<double> tmp_b = local_B_;

  world.send(left, 0, tmp_a);
  world.send(up, 1, tmp_b);
  world.recv(right, 0, local_A_);
  world.recv(down, 1, local_B_);
}

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  mpi::environment env;
  mpi::communicator world;

  rank_ = world.rank();
  size_ = world.size();

  grid_size_ = static_cast<int>(std::sqrt(size_));
  if (grid_size_ * grid_size_ != size_) {
    if (rank_ == 0) {
      std::cerr << "Number of processes must be a perfect square" << std::endl;
    }
    return false;
  }

  if (N_ % grid_size_ != 0) {
    if (rank_ == 0) {
      std::cerr << "Matrix size must be divisible by grid size" << std::endl;
    }
    return false;
  }

  block_size_ = N_ / grid_size_;
  local_size_ = block_size_ * block_size_;

  local_A_.resize(local_size_);
  local_B_.resize(local_size_);
  local_C_.resize(local_size_, 0);

  std::vector<double> tmp_a(local_size_);
  std::vector<double> tmp_b(local_size_);

  if (rank_ == 0) {
    for (int i = 0; i < grid_size_; ++i) {
      for (int j = 0; j < grid_size_; ++j) {
        int proc = i * grid_size_ + j;
        for (int bi = 0; bi < block_size_; ++bi) {
          for (int bj = 0; bj < block_size_; ++bj) {
            tmp_a[bi * block_size_ + bj] = A_[(i * block_size_ + bi) * N_ + (j * block_size_ + bj)];
            tmp_b[bi * block_size_ + bj] = B_[(i * block_size_ + bi) * N_ + (j * block_size_ + bj)];
          }
        }
        if (proc == 0) {
          local_A_ = tmp_a;
          local_B_ = tmp_b;
        } else {
          world.send(proc, 0, tmp_a);
          world.send(proc, 1, tmp_b);
        }
      }
    }
  } else {
    world.recv(0, 0, local_A_);
    world.recv(0, 1, local_B_);
  }

  InitialShift();
  for (int iter = 0; iter < grid_size_; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }

  if (rank_ == 0) {
    for (int i = 0; i < grid_size_; ++i) {
      for (int j = 0; j < grid_size_; ++j) {
        int proc = i * grid_size_ + j;
        if (proc == 0) {
          for (int bi = 0; bi < block_size_; ++bi) {
            for (int bj = 0; bj < block_size_; ++bj) {
              C_[(i * block_size_ + bi) * N_ + (j * block_size_ + bj)] = local_C_[bi * block_size_ + bj];
            }
          }
        } else {
          std::vector<double> tmp(local_size_);
          world.recv(proc, 0, tmp);
          for (int bi = 0; bi < block_size_; ++bi) {
            for (int bj = 0; bj < block_size_; ++bj) {
              C_[(i * block_size_ + bi) * N_ + (j * block_size_ + bj)] = tmp[bi * block_size_ + bj];
            }
          }
        }
      }
    }
  } else {
    world.send(0, 0, local_C_);
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
