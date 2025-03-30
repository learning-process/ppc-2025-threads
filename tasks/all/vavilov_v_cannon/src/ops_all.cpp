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
  int b_dest = (((row - col + grid_size) % grid_size) * grid_size + col);

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  if (a_dest != rank) {
    world_.send(a_dest, 0, local_A.data(), block_size_ * block_size_);
    world_.recv(a_dest, 0, tmp_A.data(), block_size_ * block_size_);
    local_A = tmp_A;
  }
  if (b_dest != rank) {
    world_.send(b_dest, 1, local_B.data(), block_size_ * block_size_);
    world_.recv(b_dest, 1, tmp_B.data(), block_size_ * block_size_);
    local_B = tmp_B;
  }

  std::cout << "[Rank " << rank << "] After InitialShift: local_A[0] = " << local_A[0]
            << ", local_B[0] = " << local_B[0] << std::endl;
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
  int up_dest = (row == 0) ? ((grid_size - 1) * grid_size + col) : (rank - grid_size);

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  if (left_dest != rank) {
    world_.send(left_dest, 2, local_A.data(), block_size_ * block_size_);
    world_.recv(left_dest, 2, tmp_A.data(), block_size_ * block_size_);
    local_A = tmp_A;
  }

  if (up_dest != rank) {
    world_.send(up_dest, 3, local_B.data(), block_size_ * block_size_);
    world_.recv(up_dest, 3, tmp_B.data(), block_size_ * block_size_);
    local_B = tmp_B;
  }

  std::cout << "[Rank " << rank << "] After ShiftBlocks: local_A[0] = " << local_A[0] << ", local_B[0] = " << local_B[0]
            << std::endl;
}

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  mpi::environment env;

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

  std::vector<double> local_A(block_size_sq, 0);
  std::vector<double> local_B(block_size_sq, 0);
  std::vector<double> local_C(block_size_sq, 0);

  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      if (p == 0) {
        std::copy(A_.begin(), A_.begin() + block_size_sq, local_A.begin());
        std::copy(B_.begin(), B_.begin() + block_size_sq, local_B.begin());
      } else {
        world_.send(p, 0, A_.data() + p * block_size_sq, block_size_sq);
        world_.send(p, 1, B_.data() + p * block_size_sq, block_size_sq);
      }
    }
  } else {
    world_.recv(0, 0, local_A.data(), block_size_sq);
    world_.recv(0, 1, local_B.data(), block_size_sq);
  }

  std::cout << "[Rank " << rank << "] After distribution: local_A[0] = " << local_A[0]
            << ", local_B[0] = " << local_B[0] << std::endl;

  InitialShift(local_A, local_B);
  BlockMultiply(local_A, local_B, local_C);

  std::cout << "[Rank " << rank << "] After first BlockMultiply: local_C[0] = " << local_C[0] << std::endl;

  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    ShiftBlocks(local_A, local_B);
    BlockMultiply(local_A, local_B, local_C);
  }

  std::cout << "[Rank " << rank << "] After all iterations: local_C[0] = " << local_C[0] << std::endl;

  if (rank == 0) {
    std::copy(local_C.begin(), local_C.end(), C_.begin());
    for (int p = 1; p < size; ++p) {
      std::vector<double> tmp_C(block_size_sq);
      world_.recv(p, 2, tmp_C.data(), block_size_sq);
      std::copy(tmp_C.begin(), tmp_C.end(), C_.begin() + p * block_size_sq);
    }
  } else {
    world_.send(0, 2, local_C.data(), block_size_sq);
  }

  if (rank == 0) {
    std::cout << "[Rank 0] Final C_: ";
    for (int i = 0; i < std::min(10, (int)C_.size()); ++i) {
      std::cout << C_[i] << " ";
    }
    std::cout << std::endl;
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
