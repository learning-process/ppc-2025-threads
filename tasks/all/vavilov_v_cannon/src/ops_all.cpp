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

void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_A, 
                                                  std::vector<double>& local_B) {
  int rank = world_.rank();
  int row_index = rank / num_blocks_;
  int col_index = rank % num_blocks_;

  // Начальный сдвиг A
  if (row_index != 0) {
    int dest = (col_index < row_index) ? 
               rank + num_blocks_ - row_index : 
               rank - row_index;
    world_.send(dest, 0, local_A);
  }

  // Начальный сдвиг B
  if (col_index != 0) {
    int dest = (row_index < col_index) ? 
               rank + (num_blocks_ - col_index) * num_blocks_ : 
               rank - num_blocks_ * col_index;
    world_.send(dest, 1, local_B);
  }

  mpi::status status;
  if (row_index != 0 && col_index != 0) {
    world_.recv(mpi::any_source, 0, local_A);
    world_.recv(mpi::any_source, 1, local_B);
  } else if (row_index == 0 && col_index != 0) {
    world_.recv(mpi::any_source, 1, local_B);
  } else if (row_index != 0 && col_index == 0) {
    world_.recv(mpi::any_source, 0, local_A);
  }
}

void vavilov_v_cannon_all::CannonALL::BlockMultiply(const std::vector<double>& local_A, 
                                                   const std::vector<double>& local_B, 
                                                   std::vector<double>& local_C) {
#pragma omp parallel for collapse(2)
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

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, 
                                                 std::vector<double>& local_B) {
  int rank = world_.rank();
  int row_index = rank / num_blocks_;
  int col_index = rank % num_blocks_;

  // Сдвиг A влево
  if (rank == row_index * num_blocks_) {
    world_.send((row_index + 1) * num_blocks_ - 1, 0, local_A);
  } else {
    world_.send(rank - 1, 0, local_A);
  }

  // Сдвиг B вверх
  if (rank < num_blocks_) {
    world_.send(rank + (num_blocks_ - 1) * num_blocks_, 1, local_B);
  } else {
    world_.send(rank - num_blocks_, 1, local_B);
  }

  world_.recv(mpi::any_source, 0, local_A);
  world_.recv(mpi::any_source, 1, local_B);
}

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  mpi::environment env;

  int rank = world_.rank();
  int size = world_.size();

  // Проверяем, что количество процессов является квадратом
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

  // Локальные буферы
  std::vector<double> local_A(block_size_sq, 0);
  std::vector<double> local_B(block_size_sq, 0);
  std::vector<double> local_C(block_size_sq, 0);

  // Распределяем матрицы
  MPI_Scatter(A_.data(), block_size_sq, MPI_DOUBLE, local_A.data(), block_size_sq, 
             MPI_DOUBLE, 0, world_);
  MPI_Scatter(B_.data(), block_size_sq, MPI_DOUBLE, local_B.data(), block_size_sq, 
             MPI_DOUBLE, 0, world_);

  // Выполняем алгоритм Кэннона
  world_.barrier();
  InitialShift(local_A, local_B);
  BlockMultiply(local_A, local_B, local_C);

  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    ShiftBlocks(local_A, local_B);
    BlockMultiply(local_A, local_B, local_C);
  }

  // Сбор результатов
  MPI_Gather(local_C.data(), block_size_sq, MPI_DOUBLE, C_.data(), block_size_sq, 
            MPI_DOUBLE, 0, world_);

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
