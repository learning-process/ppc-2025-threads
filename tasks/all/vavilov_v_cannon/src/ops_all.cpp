#include "all/vavilov_v_cannon/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

namespace mpi = boost::mpi;

int vavilov_v_cannon_all::CannonALL::find_optimal_grid_size(int size, int N) {
  int grid = std::floor(std::sqrt(size));
  while (grid > 0) {
    if (N % grid == 0) {
      break;
    }
    --grid;
  }
  return grid > 0 ? grid : 1;
}

void vavilov_v_cannon_all::CannonALL::take_block(const std::vector<double>& matrix, double* block, int N, int K,
                                                 int block_row, int block_col) {
#pragma omp parallel for
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      block[(i * K) + j] = matrix[(((block_row * K) + i) * N) + ((block_col * K) + j)];
    }
  }
}

bool vavilov_v_cannon_all::CannonALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
    num_blocks_ = static_cast<int>(std::sqrt(N_));
    block_size_ = N_ / num_blocks_;

    auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
    A_.assign(a, a + (N_ * N_));
    B_.assign(b, b + (N_ * N_));
    C_.assign(N_ * N_, 0);
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->outputs_count[0] == task_data->inputs_count[0];
  }
  return true;
}

void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  int send_rank_A = (row * grid_size) + (col + grid_size - 1) % grid_size;
  int recv_rank_A = (row * grid_size) + (col + 1) % grid_size;

  int send_rank_B = col + (grid_size * ((row + grid_size - 1) % grid_size));
  int recv_rank_B = col + (grid_size * ((row + 1) % grid_size));

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  for (int i = 0; i < row; ++i) {
    mpi::request reqs[2];
    reqs[0] = world_.isend(send_rank_A, 0, local_A.data(), block_size_ * block_size_);
    reqs[1] = world_.irecv(recv_rank_A, 0, tmp_A.data(), block_size_ * block_size_);
    mpi::wait_all(reqs, reqs + 2);
    std::swap(local_A, tmp_A);
  }

  for (int i = 0; i < col; ++i) {
    mpi::request reqs[2];
    reqs[0] = world_.isend(send_rank_B, 1, local_B.data(), block_size_ * block_size_);
    reqs[1] = world_.irecv(recv_rank_B, 1, tmp_B.data(), block_size_ * block_size_);
    mpi::wait_all(reqs, reqs + 2);
    std::swap(local_B, tmp_B);
  }
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  // Shift A left
  int send_rank_A = (row * grid_size) + (col + grid_size - 1) % grid_size;
  int recv_rank_A = (row * grid_size) + (col + 1) % grid_size;

  // Shift B up
  int send_rank_B = col + (grid_size * ((row + grid_size - 1) % grid_size));
  int recv_rank_B = col + (grid_size * ((row + 1) % grid_size));

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  mpi::request reqs[4];
  reqs[0] = world_.isend(send_rank_A, 2, local_A.data(), block_size_ * block_size_);
  reqs[1] = world_.irecv(recv_rank_A, 2, tmp_A.data(), block_size_ * block_size_);
  reqs[2] = world_.isend(send_rank_B, 3, local_B.data(), block_size_ * block_size_);
  reqs[3] = world_.irecv(recv_rank_B, 3, tmp_B.data(), block_size_ * block_size_);

  mpi::wait_all(reqs, reqs + 4);
  std::swap(local_A, tmp_A);
  std::swap(local_B, tmp_B);
}

void vavilov_v_cannon_all::CannonALL::BlockMultiply(const std::vector<double>& local_A,
                                                    const std::vector<double>& local_B, std::vector<double>& local_C) {
#pragma omp parallel for
  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      double temp = 0.0;
      for (int k = 0; k < block_size_; ++k) {
        temp += local_A[(i * block_size_) + k] * local_B[(k * block_size_) + j];
      }
      local_C[(i * block_size_) + j] += temp;
    }
  }
}
/*
void vavilov_v_cannon_all::CannonALL::InitialShiftone() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;

#pragma omp parallel for
  for (int bi = 0; bi < num_blocks_; ++bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int src_row = (bi + bj) % num_blocks_;
      int src_col = (bj + bi) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
          A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
        }
      }
    }
  }
}
void vavilov_v_cannon_all::CannonALL::BlockMultiplyone() {
#pragma omp parallel for
  for (int bi = 0; bi < num_blocks_; ++bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          double temp = 0.0;
          for (int k = 0; k < block_size_; ++k) {
            int row = (bi * block_size_) + i;
            int col = (bj * block_size_) + j;
            int k_idx = (bj * block_size_) + k;
            int k_row = (bi * block_size_) + k;
            temp += A_[(row * N_) + k_idx] * B_[(k_row * N_) + col];
          }
          C_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] += temp;
        }
      }
    }
  }
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocksone() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;

#pragma omp parallel for
  for (int bi = 0; bi < num_blocks_; ++bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int src_row = (bi + 1) % num_blocks_;
      int src_col = (bj + 1) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
          A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
        }
      }
    }
  }
}
*/
/*
bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  mpi::broadcast(world_, N_, 0);

  num_blocks_ = find_optimal_grid_size(size, N_);
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;

  int active_procs = num_blocks_ * num_blocks_;
  mpi::communicator active_world = world_.split(rank < active_procs ? 0 : MPI_UNDEFINED);
  if (rank >= active_procs) {
    return true;
  }

  rank = active_world.rank();
  size = active_world.size();

  if (num_blocks_ == 1 && size > 1) {
    num_blocks_ = static_cast<int>(std::sqrt(N_));
    if (rank == 0) {
      InitialShiftone();
      for (int iter = 0; iter < num_blocks_; ++iter) {
        BlockMultiplyone();
        ShiftBlocksone();
      }
    }
    return true;
  }

  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(block_size_sq);
  std::vector<double> local_C(block_size_sq, 0);

  std::vector<double> scatter_A;
  std::vector<double> scatter_B;
  if (rank == 0) {
    scatter_A.resize(active_procs * block_size_sq);
    scatter_B.resize(active_procs * block_size_sq);
    int index = 0;
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        take_block(A_, scatter_A.data() + index, N_, block_size_, block_row, block_col);
        take_block(B_, scatter_B.data() + index, N_, block_size_, block_row, block_col);
        index += block_size_sq;
      }
    }
  }

  mpi::scatter(active_world, scatter_A.data(), local_A.data(), block_size_sq, 0);
  mpi::scatter(active_world, scatter_B.data(), local_B.data(), block_size_sq, 0);

  InitialShift(local_A, local_B);

  BlockMultiply(local_A, local_B, local_C);
  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    ShiftBlocks(local_A, local_B);
    BlockMultiply(local_A, local_B, local_C);
  }

  std::vector<double> tmp_C;
  if (rank == 0) {
    tmp_C.resize(active_procs * block_size_sq);
  }
  mpi::gather(active_world, local_C.data(), block_size_sq, tmp_C.data(), 0);

  if (rank == 0) {
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        int block_rank = (block_row * num_blocks_) + block_col;
        int block_index = block_rank * block_size_sq;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            int global_row = (block_row * block_size_) + i;
            int global_col = (block_col * block_size_) + j;
            C_[(global_row * N_) + global_col] = tmp_C[block_index + (i * block_size_) + j];
          }
        }
      }
    }
  }

  return true;
}
*/

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  mpi::broadcast(world_, N_, 0);

  num_blocks_ = find_optimal_grid_size(size, N_);
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;

  int active_procs = num_blocks_ * num_blocks_;
  mpi::communicator active_world = world_.split(rank < active_procs ? 0 : MPI_UNDEFINED);
  if (rank >= active_procs) {
    return true;
  }

  rank = active_world.rank();
  size = active_world.size();

  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(block_size_sq);
  std::vector<double> local_C(block_size_sq, 0);

  // Раздача матриц A и B
  std::vector<double> scatter_A;
  std::vector<double> scatter_B;
  if (rank == 0) {
    scatter_A.resize(active_procs * block_size_sq);
    scatter_B.resize(active_procs * block_size_sq);
    int index = 0;
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        take_block(A_, scatter_A.data() + index, N_, block_size_, block_row, block_col);
        take_block(B_, scatter_B.data() + index, N_, block_size_, block_row, block_col);
        index += block_size_sq;
      }
    }
    mpi::scatter(active_world, scatter_A, local_A.data(), block_size_sq, 0);
    mpi::scatter(active_world, scatter_B, local_B.data(), block_size_sq, 0);
  } else {
    mpi::scatter(active_world, local_A.data(), block_size_sq, 0);
    mpi::scatter(active_world, local_B.data(), block_size_sq, 0);
  }

  int row_index = rank / num_blocks_;
  int col_index = rank % num_blocks_;

  // Начальное выравнивание
  mpi::request reqs[4];
  int req_count = 0;
  if (row_index != 0) {
    int dest_rank_A = (col_index < row_index) ? rank + num_blocks_ - row_index : rank - row_index;
    reqs[req_count++] = active_world.isend(dest_rank_A, 0, local_A.data(), block_size_sq);
  }
  if (col_index != 0) {
    int dest_rank_B =
        (row_index < col_index) ? rank + (num_blocks_ - col_index) * num_blocks_ : rank - num_blocks_ * col_index;
    reqs[req_count++] = active_world.isend(dest_rank_B, 1, local_B.data(), block_size_sq);
  }
  if (row_index != 0 && col_index != 0) {
    reqs[req_count++] = active_world.irecv(mpi::any_source, 0, local_A.data(), block_size_sq);
    reqs[req_count++] = active_world.irecv(mpi::any_source, 1, local_B.data(), block_size_sq);
  } else if (row_index == 0 && col_index != 0) {
    reqs[req_count++] = active_world.irecv(mpi::any_source, 1, local_B.data(), block_size_sq);
  } else if (row_index != 0 && col_index == 0) {
    reqs[req_count++] = active_world.irecv(mpi::any_source, 0, local_A.data(), block_size_sq);
  }
  mpi::wait_all(reqs, reqs + req_count);

  // Первое умножение
  BlockMultiply(local_A, local_B, local_C);

  // Основной цикл
  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    req_count = 0;
    int dest_rank_A = (rank == row_index * num_blocks_) ? (row_index + 1) * num_blocks_ - 1 : rank - 1;
    reqs[req_count++] = active_world.isend(dest_rank_A, 0, local_A.data(), block_size_sq);
    reqs[req_count++] = active_world.irecv(mpi::any_source, 0, local_A.data(), block_size_sq);

    int dest_rank_B = (rank < num_blocks_) ? rank + (num_blocks_ - 1) * num_blocks_ : rank - num_blocks_;
    reqs[req_count++] = active_world.isend(dest_rank_B, 1, local_B.data(), block_size_sq);
    reqs[req_count++] = active_world.irecv(mpi::any_source, 1, local_B.data(), block_size_sq);

    mpi::wait_all(reqs, reqs + req_count);

    BlockMultiply(local_A, local_B, local_C);
  }

  std::vector<double> tmp_C;
  if (rank == 0) {
    tmp_C.resize(active_procs * block_size_sq);
  }
  mpi::gather(active_world, local_C.data(), block_size_sq, tmp_C.data(), 0);

  if (rank == 0) {
#pragma omp parallel for
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        int block_rank = (block_row * num_blocks_) + block_col;
        int block_index = block_rank * block_size_sq;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            int global_row = (block_row * block_size_) + i;
            int global_col = (block_col * block_size_) + j;
            C_[(global_row * N_) + global_col] = tmp_C[block_index + (i * block_size_) + j];
          }
        }
      }
    }
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
