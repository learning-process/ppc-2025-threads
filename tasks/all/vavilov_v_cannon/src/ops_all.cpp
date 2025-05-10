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

int vavilov_v_cannon_all::CannonALL::find_compatible_q(int size, int N) {
  int q = std::floor(std::sqrt(size));
  while (q > 0) {
    if (N % q == 0) {
      break;
    }
    --q;
  }
  return q > 0 ? q : 1;
}

void vavilov_v_cannon_all::CannonALL::extract_block(const std::vector<double>& matrix, double* block, int N, int K,
                                                    int block_row, int block_col) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      block[i * K + j] = matrix[(block_row * K + i) * N + (block_col * K + j)];
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

  int send_rank_A = row * grid_size + (col + grid_size - 1) % grid_size;
  int recv_rank_A = row * grid_size + (col + 1) % grid_size;

  int send_rank_B = col + grid_size * ((row + grid_size - 1) % grid_size);
  int recv_rank_B = col + grid_size * ((row + 1) % grid_size);

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
  int send_rank_A = row * grid_size + (col + grid_size - 1) % grid_size;
  int recv_rank_A = row * grid_size + (col + 1) % grid_size;

  // Shift B up
  int send_rank_B = col + grid_size * ((row + grid_size - 1) % grid_size);
  int recv_rank_B = col + grid_size * ((row + 1) % grid_size);

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
        temp += local_A[i * block_size_ + k] * local_B[k * block_size_ + j];
      }
      local_C[i * block_size_ + j] += temp;
    }
  }
}
/*
bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  mpi::broadcast(world_, N_, 0);

  // Find compatible grid size
  num_blocks_ = find_compatible_q(size, N_);
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;

  // Create sub-communicator for active processes
  int active_procs = num_blocks_ * num_blocks_;
  mpi::communicator active_world = world_.split(rank < active_procs ? 0 : MPI_UNDEFINED);
  if (rank >= active_procs) {
    return true;
  }

  // Update rank and size for the new communicator
  rank = active_world.rank();
  size = active_world.size();

  // Initialize local matrices
  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(block_size_sq);
  std::vector<double> local_C(block_size_sq, 0);

  // Scatter matrices A and B
  std::vector<double> scatter_A;
  std::vector<double> scatter_B;
  if (rank == 0) {
    scatter_A.resize(active_procs * block_size_sq);
    scatter_B.resize(active_procs * block_size_sq);
    int index = 0;
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        extract_block(A_, scatter_A.data() + index, N_, block_size_, block_row, block_col);
        extract_block(B_, scatter_B.data() + index, N_, block_size_, block_row, block_col);
        index += block_size_sq;
      }
    }
  }

  mpi::scatter(active_world, scatter_A.data(), local_A.data(), block_size_sq, 0);
  mpi::scatter(active_world, scatter_B.data(), local_B.data(), block_size_sq, 0);

  // Perform initial alignment
  InitialShift(local_A, local_B);

  // Main computation loop
  BlockMultiply(local_A, local_B, local_C);
  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    ShiftBlocks(local_A, local_B);
    BlockMultiply(local_A, local_B, local_C);
  }

  // Gather results
  std::vector<double> tmp_C;
  if (rank == 0) {
    tmp_C.resize(active_procs * block_size_sq);
  }
  mpi::gather(active_world, local_C.data(), block_size_sq, tmp_C.data(), 0);

  // Rearrange result into C_
  if (rank == 0) {
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        int block_rank = block_row * num_blocks_ + block_col;
        int block_index = block_rank * block_size_sq;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            int global_row = block_row * block_size_ + i;
            int global_col = block_col * block_size_ + j;
            C_[global_row * N_ + global_col] = tmp_C[block_index + i * block_size_ + j];
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

  // Находим совместимый размер сетки
  num_blocks_ = find_compatible_q(size, N_);
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;

  // Создаём субкоммуникатор для активных процессов
  int active_procs = num_blocks_ * num_blocks_;
  mpi::communicator active_world = world_.split(rank < active_procs ? 0 : MPI_UNDEFINED);
  if (rank >= active_procs) {
    return true;
  }

  // Инициализируем локальные матрицы
  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(block_size_sq);
  std::vector<double> local_C(block_size_sq, 0);

  // Раздача матриц A и B
  if (rank == 0) {
    std::vector<double> scatter_A(active_procs * block_size_sq);
    std::vector<double> scatter_B(active_procs * block_size_sq);
    int index = 0;
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        extract_block(A_, scatter_A.data() + index, N_, block_size_, block_row, block_col);
        extract_block(B_, scatter_B.data() + index, N_, block_size_, block_row, block_col);
        index += block_size_sq;
      }
    }
    mpi::scatter(active_world, scatter_A.data(), local_A.data(), block_size_sq, 0);
    mpi::scatter(active_world, scatter_B.data(), local_B.data(), block_size_sq, 0);
  } else {
    mpi::scatter(active_world, nullptr, local_A.data(), block_size_sq, 0);
    mpi::scatter(active_world, nullptr, local_B.data(), block_size_sq, 0);
  }

  // Начальное выравнивание
  InitialShift(local_A, local_B);

  // Основной цикл вычислений
  for (int iter = 0; iter < num_blocks_; ++iter) {
    // Выполняем умножение блоков
    BlockMultiply(local_A, local_B, local_C);
    if (iter < num_blocks_ - 1) {
      // Сдвигаем блоки для следующей итерации
      ShiftBlocks(local_A, local_B);
    }
  }

  // Сбор результатов
  std::vector<double> tmp_C;
  if (rank == 0) {
    tmp_C.resize(active_procs * block_size_sq);
  }
  mpi::gather(active_world, local_C.data(), block_size_sq, tmp_C.data(), 0);

  // Формирование итоговой матрицы C_
  if (rank == 0) {
    for (int block_row = 0; block_row < num_blocks_; ++block_row) {
      for (int block_col = 0; block_col < num_blocks_; ++block_col) {
        int block_rank = block_row * num_blocks_ + block_col;
        int block_index = block_rank * block_size_sq;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            int global_row = block_row * block_size_ + i;
            int global_col = block_col * block_size_ + j;
            C_[global_row * N_ + global_col] = tmp_C[block_index + i * block_size_ + j];
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
