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
/*
void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  int a_dest = row * grid_size + (col - row + grid_size) % grid_size;
  int a_src = row * grid_size + (col + row) % grid_size;

  int b_dest = ((row - col + grid_size) % grid_size) * grid_size + col;
  int b_src = ((row + col) % grid_size) * grid_size + col;

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  mpi::request reqs[4];
  if (a_dest != rank) {
    reqs[0] = world_.isend(a_dest, 0, local_A.data(), block_size_ * block_size_);
    reqs[1] = world_.irecv(a_src, 0, tmp_A.data(), block_size_ * block_size_);
  } else {
    tmp_A = local_A;
  }

  if (b_dest != rank) {
    reqs[2] = world_.isend(b_dest, 1, local_B.data(), block_size_ * block_size_);
    reqs[3] = world_.irecv(b_src, 1, tmp_B.data(), block_size_ * block_size_);
  } else {
    tmp_B = local_B;
  }

  if (a_dest != rank || b_dest != rank) {
    mpi::wait_all(reqs, reqs + 4);
  }

  local_A = tmp_A;
  local_B = tmp_B;
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  int left_dest = row * grid_size + (col == 0 ? grid_size - 1 : col - 1);
  int left_src = row * grid_size + (col == grid_size - 1 ? 0 : col + 1);
  int up_dest = (row == 0 ? grid_size - 1 : row - 1) * grid_size + col;
  int up_src = (row == grid_size - 1 ? 0 : row + 1) * grid_size + col;

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  mpi::request reqs[4];
  if (left_dest != rank) {
    reqs[0] = world_.isend(left_dest, 2, local_A.data(), block_size_ * block_size_);
    reqs[1] = world_.irecv(left_src, 2, tmp_A.data(), block_size_ * block_size_);
  } else {
    tmp_A = local_A;
  }

  if (up_dest != rank) {
    reqs[2] = world_.isend(up_dest, 3, local_B.data(), block_size_ * block_size_);
    reqs[3] = world_.irecv(up_src, 3, tmp_B.data(), block_size_ * block_size_);
  } else {
    tmp_B = local_B;
  }

  if (left_dest != rank || up_dest != rank) {
    mpi::wait_all(reqs, reqs + 4);
  }

  local_A = tmp_A;
  local_B = tmp_B;
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

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank == 0) {
    std::cout << "Running with " << size << " processes" << std::endl;
  }

  int grid_size = static_cast<int>(std::sqrt(size));
  int active_procs = grid_size * grid_size;

  if (rank >= active_procs) {
    if (rank == 0) {
      std::cout << "Rank " << rank << " is inactive (active_procs = " << active_procs << ")" << std::endl;
    }
    return true;
  }

  mpi::communicator active_world = world_.split(rank < active_procs ? 0 : MPI_UNDEFINED);
  if (rank >= active_procs) {
    return true;
  }

  if (rank == 0) {
    std::cout << "Active processes: " << active_procs << ", grid_size: " << grid_size << std::endl;
  }

  if (N_ % grid_size != 0) {
    if (rank == 0) {
      std::cerr << "Matrix size (" << N_ << ") must be divisible by grid size (" << grid_size << ")" << std::endl;
    }
    return false;
  }

  num_blocks_ = grid_size;
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;
  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(block_size_sq);
  std::vector<double> local_C(block_size_sq, 0);

  if (active_procs == 1) {
    if (rank == 0) {
      std::cout << "Single process mode: performing sequential matrix multiplication" << std::endl;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          local_A[i * block_size_ + j] = A_[i * N_ + j];
          local_B[i * block_size_ + j] = B_[i * N_ + j];
        }
      }
      // Выполняем умножение
      BlockMultiply(local_A, local_B, local_C);
      // Копируем результат в C_
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          C_[i * N_ + j] = local_C[i * block_size_ + j];
        }
      }
    }
    return true;
  }

  if (rank == 0) {
    std::cout << "Rank 0: Scattering matrices A and B" << std::endl;
    std::vector<double> tmp_A(active_procs * block_size_sq);
    std::vector<double> tmp_B(active_procs * block_size_sq);
    for (int p = 0; p < active_procs; ++p) {
      int row_p = p / grid_size;
      int col_p = p % grid_size;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          tmp_A[p * block_size_sq + i * block_size_ + j] =
              A_[(row_p * block_size_ + i) * N_ + (col_p * block_size_ + j)];
          tmp_B[p * block_size_sq + i * block_size_ + j] =
              B_[(row_p * block_size_ + i) * N_ + (col_p * block_size_ + j)];
        }
      }
    }
    mpi::scatter(active_world, tmp_A.data(), local_A.data(), block_size_sq, 0);
    mpi::scatter(active_world, tmp_B.data(), local_B.data(), block_size_sq, 0);
  } else {
    mpi::scatter(active_world, local_A.data(), block_size_sq, 0);
    mpi::scatter(active_world, local_B.data(), block_size_sq, 0);
  }

  int row_index = rank / num_blocks_;
  int col_index = rank % num_blocks_;
  std::vector<mpi::request> reqs;

  if (num_blocks_ > 1) {
    if (row_index != col_index) {
      int dest_A = row_index * num_blocks_ + (col_index + row_index) % num_blocks_;
      int src_A = row_index * num_blocks_ + (col_index - row_index + num_blocks_) % num_blocks_;
      reqs.push_back(active_world.isend(dest_A, 0, local_A.data(), block_size_sq));
      reqs.push_back(active_world.irecv(src_A, 0, local_A.data(), block_size_sq));
    }
    if (row_index != col_index) {
      int dest_B = ((row_index + col_index) % num_blocks_) * num_blocks_ + col_index;
      int src_B = ((row_index - col_index + num_blocks_) % num_blocks_) * num_blocks_ + col_index;
      reqs.push_back(active_world.isend(dest_B, 1, local_B.data(), block_size_sq));
      reqs.push_back(active_world.irecv(src_B, 1, local_B.data(), block_size_sq));
    }
    if (!reqs.empty()) {
      mpi::wait_all(reqs.begin(), reqs.end());
      reqs.clear();
    }
  }

  active_world.barrier();
  BlockMultiply(local_A, local_B, local_C);

  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    int dest_A = row_index * num_blocks_ + (col_index == 0 ? num_blocks_ - 1 : col_index - 1);
    int src_A = row_index * num_blocks_ + (col_index == num_blocks_ - 1 ? 0 : col_index + 1);
    reqs.push_back(active_world.isend(dest_A, 2, local_A.data(), block_size_sq));
    reqs.push_back(active_world.irecv(src_A, 2, local_A.data(), block_size_sq));

    int dest_B = (row_index == 0 ? num_blocks_ - 1 : row_index - 1) * num_blocks_ + col_index;
    int src_B = (row_index == num_blocks_ - 1 ? 0 : row_index + 1) * num_blocks_ + col_index;
    reqs.push_back(active_world.isend(dest_B, 3, local_B.data(), block_size_sq));
    reqs.push_back(active_world.irecv(src_B, 3, local_B.data(), block_size_sq));

    mpi::wait_all(reqs.begin(), reqs.end());
    reqs.clear();

    BlockMultiply(local_A, local_B, local_C);
  }

  if (rank == 0) {
    std::cout << "Rank 0: Gathering results" << std::endl;
    std::vector<double> tmp_C(active_procs * block_size_sq);
    mpi::gather(active_world, local_C.data(), block_size_sq, tmp_C.data(), 0);
    for (int p = 0; p < active_procs; ++p) {
      int row_p = p / grid_size;
      int col_p = p % grid_size;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          C_[(row_p * block_size_ + i) * N_ + (col_p * block_size_ + j)] =
              tmp_C[p * block_size_sq + i * block_size_ + j];
        }
      }
    }
  } else {
    mpi::gather(active_world, local_C.data(), block_size_sq, 0);
  }

  return true;
}
*/
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

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank == 0) {
      std::cout << "Running Cannon's algorithm with " << size << " processes\n";
  }

  int grid_rows = 1;
  int grid_cols = size;
  for (int i = static_cast<int>(std::sqrt(size)); i >= 1; --i) {
    if (size % i == 0) {
      grid_rows = i;
      grid_cols = size / i;
      break;
    }
  }

  if (N_ % grid_rows != 0 || N_ % grid_cols != 0) {
    if (rank == 0) {
    std::cerr << "Matrix size " << N_ << " must be divisible by " << grid_rows << " and " << grid_cols << "\n";
    }
    return false;
  }

  const int block_rows = N_ / grid_rows;
  const int block_cols = N_ / grid_cols;
  const int block_size_A = block_rows * block_cols;
  const int block_size_B = block_cols * block_rows;
  const int block_size_C = block_rows * block_rows;

  std::vector<double> local_A(block_size_A);
  std::vector<double> local_B(block_size_B);
  std::vector<double> local_C(block_size_C, 0.0);
  // Распределение данных
  if (rank == 0) {
    std::vector<double> tmp_A(size * block_size_A);
    std::vector<double> tmp_B(size * block_size_B);

    for (int p = 0; p < size; ++p) {
      int row = p / grid_cols;
      int col = p % grid_cols;

      // Распределение A
      for (int i = 0; i < block_rows; ++i) {
        for (int j = 0; j < block_cols; ++j) {
          tmp_A[p * block_size_A + i * block_cols + j] = A_[(row * block_rows + i) * N_ + (col * block_cols + j)];
        }
      }

      // Распределение B (транспонированное)
      for (int i = 0; i < block_cols; ++i) {
        for (int j = 0; j < block_rows; ++j) {
          tmp_B[p * block_size_B + i * block_rows + j] = B_[(col * block_cols + i) * N_ + (row * block_rows + j)];
        }
      }
    }

    mpi::scatter(world_, tmp_A.data(), block_size_A, local_A.data(), 0);
    mpi::scatter(world_, tmp_B.data(), block_size_B, local_B.data(), 0);
  } else {
    mpi::scatter(world_, local_A.data(), block_size_A, 0);
    mpi::scatter(world_, local_B.data(), block_size_B, 0);
  }

  // Начальное выравнивание
  int row = rank / grid_cols;
  int col = rank % grid_cols;

  // Сдвиг A влево на 'row' позиций
  int dest_A = row * grid_cols + (col + row) % grid_cols;
  int src_A = row * grid_cols + (col - row + grid_cols) % grid_cols;
  world_.sendrecv(local_A.data(), block_size_A, dest_A, 0, local_A.data(), block_size_A, src_A, 0);

  // Сдвиг B вверх на 'col' позиций
  int dest_B = ((row + col) % grid_rows) * grid_cols + col;
  int src_B = ((row - col + grid_rows) % grid_rows) * grid_cols + col;
  world_.sendrecv(local_B.data(), block_size_B, dest_B, 1, local_B.data(), block_size_B, src_B, 1);

  // Основной цикл умножения
  for (int step = 0; step < grid_cols; ++step) {
    // Локальное умножение
    for (int i = 0; i < block_rows; ++i) {
      for (int j = 0; j < block_rows; ++j) {
        double sum = 0.0;
        for (int k = 0; k < block_cols; ++k) {
          sum += local_A[i * block_cols + k] * local_B[k * block_rows + j];
        }
        local_C[i * block_rows + j] += sum;
      }
    }

    if (step < grid_cols - 1) {
      // Циклический сдвиг A влево на 1
      int left_neigh = row * grid_cols + (col - 1 + grid_cols) % grid_cols;
      int right_neigh = row * grid_cols + (col + 1) % grid_cols;
      world_.sendrecv(local_A.data(), block_size_A, left_neigh, 2, local_A.data(), block_size_A, right_neigh, 2);

      // Циклический сдвиг B вверх на 1
      int up_neigh = ((row - 1 + grid_rows) % grid_rows) * grid_cols + col;
      int down_neigh = ((row + 1) % grid_rows) * grid_cols + col;
      world_.sendrecv(local_B.data(), block_size_B, up_neigh, 3, local_B.data(), block_size_B, down_neigh, 3);
    }
  }
  // Сбор результатов
  if (rank == 0) {
    std::vector<double> gathered_C(N_ * N_);

    // Копируем данные от rank 0
    for (int i = 0; i < block_rows; ++i) {
      for (int j = 0; j < block_rows; ++j) {
        gathered_C[i * N_ + j] = local_C[i * block_rows + j];
      }
    }

    // Получаем данные от других процессов
    for (int p = 1; p < size; ++p) {
      std::vector<double> temp(block_size_C);
      world_.recv(p, 4, temp.data(), block_size_C);
      int row = p / grid_cols;
      int col = p % grid_cols;
      for (int i = 0; i < block_rows; ++i) {
        for (int j = 0; j < block_rows; ++j) {
          gathered_C[(row * block_rows + i) * N_ + (col * block_rows + j)] = temp[i * block_rows + j];
        }
      }
    }

    std::copy(gathered_C.begin(), gathered_C.end(), C_.begin());
  } else {
    world_.send(0, 4, local_C.data(), block_size_C);
  }

  return true;
 }

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
