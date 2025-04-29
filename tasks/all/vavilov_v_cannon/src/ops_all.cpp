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
/*
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
  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(block_size_sq);
  std::vector<double> local_C(block_size_sq, 0);

  if (rank == 0) {
    std::vector<double> tmp_A(size * block_size_sq);
    std::vector<double> tmp_B(size * block_size_sq);
    for (int p = 0; p < size; ++p) {
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
    mpi::scatter(world_, tmp_A.data(), local_A.data(), block_size_sq, 0);
    mpi::scatter(world_, tmp_B.data(), local_B.data(), block_size_sq, 0);
  } else {
    mpi::scatter(world_, local_A.data(), block_size_sq, 0);
    mpi::scatter(world_, local_B.data(), block_size_sq, 0);
  }

  int row_index = rank / num_blocks_;
  int col_index = rank % num_blocks_;
  std::vector<mpi::request> reqs;

  if (row_index != 0) {
    int dest_A = row_index * num_blocks_ + (col_index + row_index) % num_blocks_;
    reqs.push_back(world_.isend(dest_A, 0, local_A.data(), block_size_sq));
    std::cout << "Rank " << rank << " sending A to " << dest_A << std::endl;
  }
  if (col_index != 0) {
    int dest_B = ((row_index + col_index) % num_blocks_) * num_blocks_ + col_index;
    reqs.push_back(world_.isend(dest_B, 1, local_B.data(), block_size_sq));
    std::cout << "Rank " << rank << " sending B to " << dest_B << std::endl;
  }
  if (row_index != 0 && col_index != 0) {
    reqs.push_back(world_.irecv(mpi::any_source, 0, local_A.data(), block_size_sq));
    reqs.push_back(world_.irecv(mpi::any_source, 1, local_B.data(), block_size_sq));
  } else if (row_index != 0) {
    reqs.push_back(world_.irecv(mpi::any_source, 0, local_A.data(), block_size_sq));
  } else if (col_index != 0) {
    reqs.push_back(world_.irecv(mpi::any_source, 1, local_B.data(), block_size_sq));
  }
  if (!reqs.empty()) {
    mpi::wait_all(reqs.begin(), reqs.end());
    reqs.clear();
  }

  world_.barrier();
  BlockMultiply(local_A, local_B, local_C);

  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    int dest_A, dest_B;

    dest_A = row_index * num_blocks_ + (col_index == 0 ? num_blocks_ - 1 : col_index - 1);
    reqs.push_back(world_.isend(dest_A, 2, local_A.data(), block_size_sq));
    std::cout << "Rank " << rank << " iter " << iter << " sending A to " << dest_A << std::endl;

    dest_B = (row_index == 0 ? num_blocks_ - 1 : row_index - 1) * num_blocks_ + col_index;
    reqs.push_back(world_.isend(dest_B, 3, local_B.data(), block_size_sq));
    std::cout << "Rank " << rank << " iter " << iter << " sending B to " << dest_B << std::endl;

    reqs.push_back(world_.irecv(mpi::any_source, 2, local_A.data(), block_size_sq));
    reqs.push_back(world_.irecv(mpi::any_source, 3, local_B.data(), block_size_sq));

    mpi::wait_all(reqs.begin(), reqs.end());
    reqs.clear();

    BlockMultiply(local_A, local_B, local_C);
  }

  if (rank == 0) {
    std::vector<double> tmp_C(size * block_size_sq);
    mpi::gather(world_, local_C.data(), block_size_sq, tmp_C.data(), 0);
    for (int p = 0; p < size; ++p) {
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
    mpi::gather(world_, local_C.data(), block_size_sq, 0);
  }

  return true;
}
*/
bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  block_size_ = N_ / size;
  int block_size_sq = block_size_ * N_;
  std::vector<double> local_A(block_size_sq);
  std::vector<double> local_B(N_ * N_);
  std::vector<double> local_C(block_size_sq, 0);

  if (rank == 0) {
    std::vector<double> tmp_A(size * block_size_sq);
    for (int p = 0; p < size; ++p) {
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < N_; ++j) {
          tmp_A[p * block_size_sq + i * N_ + j] = A_[(p * block_size_ + i) * N_ + j];
        }
      }
    }
    mpi::scatter(world_, tmp_A.data(), local_A.data(), block_size_sq, 0);
    mpi::bcast(world_, B_.data(), N_ * N_, 0);
  } else {
    mpi::scatter(world_, local_A.data(), block_size_sq, 0);
    mpi::bcast(world_, local_B.data(), N_ * N_, 0);
  }

  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < N_; ++j) {
      double temp = 0.0;
      for (int k = 0; k < N_; ++k) {
        temp += local_A[i * N_ + k] * local_B[k * N_ + j];
      }
      local_C[i * N_ + j] = temp;
    }
  }

  if (rank == 0) {
    std::vector<double> tmp_C(size * block_size_sq);
    mpi::gather(world_, local_C.data(), block_size_sq, tmp_C.data(), 0);
    for (int p = 0; p < size; ++p) {
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < N_; ++j) {
          C_[(p * block_size_ + i) * N_ + j] = tmp_C[p * block_size_sq + i * N_ + j];
        }
      }
    }
  } else {
    mpi::gather(world_, local_C.data(), block_size_sq, 0);
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
