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
  int size = world_.size();
  int grid_size = num_blocks_;

  std::vector<double> tmp_A = local_A;
  std::vector<double> tmp_B = local_B;

  int row = rank / grid_size;
  int col = rank % grid_size;

  // Вычисляем смещения для начального сдвига
  int a_dest = (row * grid_size + (col + row) % grid_size);
  int b_dest = (((row + col) % grid_size) * grid_size + col);

  // Создаем буферы для всех процессов
  std::vector<std::vector<double>> all_A(size, std::vector<double>(block_size_ * block_size_));
  std::vector<std::vector<double>> all_B(size, std::vector<double>(block_size_ * block_size_));

  // Собираем данные от всех процессов
  mpi::all_gather(world_, tmp_A, all_A);
  mpi::all_gather(world_, tmp_B, all_B);

  // Каждый процесс выбирает нужный блок после сдвига
  local_A = all_A[a_dest];
  local_B = all_B[b_dest];
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

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, 
                                                 std::vector<double>& local_B) {
  int rank = world_.rank();
  int size = world_.size();
  int grid_size = num_blocks_;

  std::vector<double> tmp_A = local_A;
  std::vector<double> tmp_B = local_B;

  int row = rank / grid_size;
  int col = rank % grid_size;

  int left_dest = (col == 0) ? (row * grid_size + grid_size - 1) : (rank - 1);
  int up_dest = (row == 0) ? ((grid_size - 1) * grid_size + col) : (rank - grid_size);

  std::vector<std::vector<double>> all_A(size, std::vector<double>(block_size_ * block_size_));
  std::vector<std::vector<double>> all_B(size, std::vector<double>(block_size_ * block_size_));

  mpi::all_gather(world_, tmp_A, all_A);
  mpi::all_gather(world_, tmp_B, all_B);

  local_A = all_A[left_dest];
  local_B = all_B[up_dest];
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

  // Локальные буферы
  std::vector<double> local_A(block_size_sq, 0);
  std::vector<double> local_B(block_size_sq, 0);
  std::vector<double> local_C(block_size_sq, 0);

  // Распределяем матрицы с использованием указателей на данные
  if (rank == 0) {
    mpi::scatter(world_, A_.data(), local_A.data(), block_size_sq, 0);
    mpi::scatter(world_, B_.data(), local_B.data(), block_size_sq, 0);
  } else {
    mpi::scatter(world_, static_cast<double*>(nullptr), local_A.data(), block_size_sq, 0);
    mpi::scatter(world_, static_cast<double*>(nullptr), local_B.data(), block_size_sq, 0);
  }

  InitialShift(local_A, local_B);
  BlockMultiply(local_A, local_B, local_C);
  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    ShiftBlocks(local_A, local_B);
    BlockMultiply(local_A, local_B, local_C);
  }
  // Сбор результатов
  if (rank == 0) {
    mpi::gather(world_, local_C.data(), C_.data(), block_size_sq, 0);
  } else {
    mpi::gather(world_, local_C.data(), static_cast<double*>(nullptr), block_size_sq, 0);
  }
  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
