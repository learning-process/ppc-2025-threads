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
/*
void vavilov_v_cannon_all::CannonALL::CalculateGridDimensions(int size, int& rows, int& cols) {
  // Находим такие rows и cols, чтобы rows * cols = size и rows максимально близко к cols
  rows = static_cast<int>(std::sqrt(size));
  while (size % rows != 0 && rows > 1) {
    rows--;
  }
  cols = size / rows;
}

void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int row = rank / num_cols_;
  int col = rank % num_cols_;

  int a_dest = row * num_cols_ + (col == 0 ? num_cols_ - 1 : col - 1);
  int a_src = row * num_cols_ + (col == num_cols_ - 1 ? 0 : col + 1);

  int b_dest = rank;
  int b_src = rank;

  std::vector<double> tmp_A(block_size_ * block_size_);
  std::vector<double> tmp_B(block_size_ * block_size_);

  std::vector<mpi::request> reqs;
  if (num_rows_ > 1 && b_dest != rank) {
    reqs.push_back(world_.isend(b_dest, 1, local_B.data(), block_size_ * block_size_));
    reqs.push_back(world_.irecv(b_src, 1, tmp_B.data(), block_size_ * block_size_));
  } else {
    tmp_B = local_B;
  }

  if (a_dest != rank) {
    reqs.push_back(world_.isend(a_dest, 0, local_A.data(), block_size_ * block_size_));
    reqs.push_back(world_.irecv(a_src, 0, tmp_A.data(), block_size_ * block_size_));
  } else {
    tmp_A = local_A;
  }

  if (!reqs.empty()) {
    mpi::wait_all(reqs.begin(), reqs.end());
  }

  local_A = tmp_A;
  local_B = tmp_B;
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int row = rank / num_cols_;
  int col = rank % num_cols_;

  // Сдвиг A влево
  int left_dest = row * num_cols_ + (col == 0 ? num_cols_ - 1 : col - 1);
  int left_src = row * num_cols_ + (col == num_cols_ - 1 ? 0 : col + 1);

  // Сдвиг B вверх
  int up_dest = (row == 0 ? num_rows_ - 1 : row - 1) * num_cols_ + col;
  int up_src = (row == num_rows_ - 1 ? 0 : row + 1) * num_cols_ + col;

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

  // Вычисляем размеры сетки процессов
  CalculateGridDimensions(size, num_rows_, num_cols_);
  int active_procs = num_rows_ * num_cols_;

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
    std::cout << "Active processes: " << active_procs << ", grid: " << num_rows_ << "x" << num_cols_ << std::endl;
  }

  // Распределяем N_ на все процессы
  mpi::broadcast(world_, N_, 0);

  // Проверяем, делится ли размер матрицы на размеры блоков
  block_size_ = N_ / num_rows_;
  if (N_ % num_rows_ != 0 || N_ % num_cols_ != 0) {
    if (rank == 0) {
      std::cerr << "Matrix size (" << N_ << ") must be divisible by grid dimensions (" << num_rows_ << "x" << num_cols_
                << ")" << std::endl;
    }
    return false;
  }

  // Проверяем корректность block_size_
  if (block_size_ <= 0) {
    if (rank == 0) {
      std::cerr << "Invalid block_size: " << block_size_ << std::endl;
    }
    return false;
  }

  int block_size_sq = block_size_ * block_size_;
  if (block_size_sq <= 0 || block_size_sq > 1e9) {
    if (rank == 0) {
      std::cerr << "Invalid block_size_sq: " << block_size_sq << " (N_ = " << N_ << ", num_rows_ = " << num_rows_ << ")"
                << std::endl;
    }
    return false;
  }

  try {
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
        BlockMultiply(local_A, local_B, local_C);
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            C_[i * N_ + j] = local_C[i * block_size_ + j];
          }
        }
      }
      return true;
    }

    // Разбрасываем матрицы A и B
    if (rank == 0) {
      std::cout << "Rank 0: Scattering matrices A and B" << std::endl;
      std::vector<double> tmp_A(active_procs * block_size_sq);
      std::vector<double> tmp_B(active_procs * block_size_sq);
      for (int p = 0; p < active_procs; ++p) {
        int row_p = p / num_cols_;
        int col_p = p % num_cols_;
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

    // Начальный сдвиг
    InitialShift(local_A, local_B);

    active_world.barrier();
    BlockMultiply(local_A, local_B, local_C);

    // Для сетки 1xN нужен дополнительный сдвиг A
    if (num_rows_ == 1) {
      ShiftBlocks(local_A, local_B);
      BlockMultiply(local_A, local_B, local_C);
    } else {
      for (int iter = 0; iter < num_rows_ - 1; ++iter) {
        ShiftBlocks(local_A, local_B);
        BlockMultiply(local_A, local_B, local_C);
      }
    }

    // Собираем результаты
    if (rank == 0) {
      std::cout << "Rank 0: Gathering results" << std::endl;
      std::vector<double> tmp_C(active_procs * block_size_sq);
      mpi::gather(active_world, local_C.data(), block_size_sq, tmp_C.data(), 0);
      for (int p = 0; p < active_procs; ++p) {
        int row_p = p / num_cols_;
        int col_p = p % num_cols_;
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
  } catch (const std::bad_alloc& e) {
    if (rank == 0) {
      std::cerr << "Bad allocation in RunImpl: " << e.what() << std::endl;
      std::cerr << "N_ = " << N_ << ", num_rows_ = " << num_rows_ << ", num_cols_ = " << num_cols_
                << ", block_size_ = " << block_size_ << std::endl;
    }
    return false;
  }

  return true;
}
*/
void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int grid_coord_x = rank / grid_size;
  int grid_coord_y = rank % grid_size;

  std::vector<double> block_buffer(block_size_ * block_size_);

  // Shift A: each row i shifts left by i positions
  int dest_rank_a = grid_coord_x * grid_size + (grid_coord_y - grid_coord_x + grid_size) % grid_size;
  int src_rank_a = grid_coord_x * grid_size + (grid_coord_y + grid_coord_x) % grid_size;

  if (dest_rank_a != rank) {
    world_.send(dest_rank_a, 0, local_A);
    world_.recv(src_rank_a, 0, local_A);
  }

  // Shift B: each column j shifts up by j positions
  int dest_rank_b = ((grid_coord_x - grid_coord_y + grid_size) % grid_size) * grid_size + grid_coord_y;
  int src_rank_b = ((grid_coord_x + grid_coord_y) % grid_size) * grid_size + grid_coord_y;

  if (dest_rank_b != rank) {
    world_.send(dest_rank_b, 1, local_B);
    world_.recv(src_rank_b, 1, local_B);
  }
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_A, std::vector<double>& local_B) {
  int rank = world_.rank();
  int grid_size = num_blocks_;

  // Shift A left
  int left_rank = (rank / grid_size) * grid_size + ((rank % grid_size) - 1 + grid_size) % grid_size;
  int right_rank = (rank / grid_size) * grid_size + ((rank % grid_size) + 1) % grid_size;

  if (left_rank != rank) {
    world_.send(left_rank, 2, local_A);
    world_.recv(right_rank, 2, local_A);
  }

  // Shift B up
  int up_rank = (((rank / grid_size) - 1 + grid_size) % grid_size) * grid_size + (rank % grid_size);
  int down_rank = (((rank / grid_size) + 1) % grid_size) * grid_size + (rank % grid_size);

  if (up_rank != rank) {
    world_.send(up_rank, 3, local_B);
    world_.recv(down_rank, 3, local_B);
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

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  if (grid_size * grid_size != world_.size()) {
    return false;
  }

  std::vector<double> local_A(block_size_ * block_size_);
  std::vector<double> local_B(block_size_ * block_size_);
  std::vector<double> local_C(block_size_ * block_size_, 0.0);

  int grid_coord_x = rank / grid_size;
  int grid_coord_y = rank % grid_size;

  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      local_A[i * block_size_ + j] = A_[(grid_coord_x * block_size_ + i) * N_ + (grid_coord_y * block_size_ + j)];
      local_B[i * block_size_ + j] = B_[(grid_coord_x * block_size_ + i) * N_ + (grid_coord_y * block_size_ + j)];
    }
  }

  InitialShift(local_A, local_B);
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply(local_A, local_B, local_C);
    ShiftBlocks(local_A, local_B);
  }

  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
        C_[(grid_coord_x * block_size_ + i) * N_ + (grid_coord_y * block_size_ + j)] = local_C[i * block_size_ + j];
      }
  }

  boost::mpi::all_reduce(world_, C_.data(), C_.size(), C_.data(), std::plus<double>());

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
