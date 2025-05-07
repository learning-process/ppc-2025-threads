#include "all/moiseev_a_mult_mat/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

bool moiseev_a_mult_mat_mpi::MultMatMPI::PreProcessingImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];

  auto *in_ptr_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *in_ptr_b = reinterpret_cast<double *>(task_data->inputs[1]);

  matrix_a_ = std::vector<double>(in_ptr_a, in_ptr_a + input_size_a);
  matrix_b_ = std::vector<double>(in_ptr_b, in_ptr_b + input_size_b);

  unsigned int output_size = task_data->outputs_count[0];
  matrix_c_ = std::vector<double>(output_size, 0.0);

  matrix_size_ = static_cast<int>(std::sqrt(input_size_a));
  block_size_ = static_cast<int>(std::sqrt(matrix_size_));
  if (matrix_size_ % block_size_ != 0) {
    block_size_ = 1;
  }
  num_blocks_ = matrix_size_ / block_size_;
  return true;
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::RunImpl() {  // NOLINT

  boost::mpi::communicator world;
  int world_size = world.size();
  int world_rank = world.rank();

  int p = static_cast<int>(std::sqrt(world_size));
  while (p > 1 && (matrix_size_ % p != 0 || (p * p) > world_size)) {
    p--;
  }
  p = std::max(p, 1);
  int active_procs = p * p;
  int block = matrix_size_ / p;

  boost::mpi::communicator active = world.split(world_rank < active_procs ? 0 : MPI_UNDEFINED, world_rank);
  bool is_active = (world_rank < active_procs);

  if (is_active) {
    int my_row = world_rank / p;
    int my_col = world_rank % p;

    std::vector<double> a_block(block * block);
    std::vector<double> b_block(block * block);
    std::vector<double> c_block(block * block, 0.0);

    if (world_rank == 0) {
      for (int proc = 0; proc < active_procs; ++proc) {
        int row = proc / p;
        int col = proc % p;
        std::vector<double> tmp(block * block);
        for (int i = 0; i < block; ++i) {
          int src = ((row * block + i) * matrix_size_) + (col * block);
          std::copy_n(&matrix_a_[src], block, &tmp[i * block]);
        }
        if (proc == 0) {
          a_block = tmp;
        } else {
          world.send(proc, 0, tmp);
        }
      }
      for (int proc = 0; proc < active_procs; ++proc) {
        int row = proc / p;
        int col = proc % p;
        std::vector<double> tmp(block * block);
        for (int i = 0; i < block; ++i) {
          int src = ((row * block + i) * matrix_size_) + (col * block);
          std::copy_n(&matrix_b_[src], block, &tmp[i * block]);
        }
        if (proc == 0) {
          b_block = tmp;
        } else {
          world.send(proc, 1, tmp);
        }
      }
    } else {
      world.recv(0, 0, a_block);
      world.recv(0, 1, b_block);
    }

    auto orig_a = a_block;
    boost::mpi::communicator row_comm = active.split(my_row, my_col);

    for (int step = 0; step < p; ++step) {
      a_block = orig_a;
      int root = (my_row + step) % p;
      broadcast(row_comm, a_block, root);

#pragma omp parallel for
      for (int i = 0; i < block; ++i) {
        for (int j = 0; j < block; ++j) {
          double sum = 0;
          for (int k = 0; k < block; ++k) {
            sum += a_block[(i * block) + k] * b_block[(k * block) + j];
          }
          c_block[(i * block) + j] += sum;
        }
      }

      int prev = (my_row - 1 + p) % p;
      int next = (my_row + 1) % p;
      active.sendrecv((prev * p) + my_col, 2, b_block, (next * p) + my_col, 2, b_block);
    }

    if (world_rank == 0) {
      for (int i = 0; i < block; ++i) {
        int dst = ((i + my_row * block) * matrix_size_) + (my_col * block);
        std::copy_n(&c_block[i * block], block, &matrix_c_[dst]);
      }
      for (int proc = 1; proc < active_procs; ++proc) {
        std::vector<double> tmp(block * block);
        world.recv(proc, 3, tmp);
        int row = proc / p;
        int col = proc % p;
        for (int i = 0; i < block; ++i) {
          int dst = ((i + row * block) * matrix_size_) + (col * block);
          std::copy_n(&tmp[i * block], block, &matrix_c_[dst]);
        }
      }
    } else {
      world.send(0, 3, c_block);
    }
  }

  broadcast(world, matrix_c_, 0);
  return true;
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(matrix_c_, out_ptr);
  return true;
}
