#include "all/moiseev_a_mult_mat/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int grid_dim = static_cast<int>(std::sqrt(world_size));
  while (grid_dim > 1 && (matrix_size_ % grid_dim != 0 || grid_dim * grid_dim > world_size)) {
    grid_dim--;
  }
  grid_dim = std::max(grid_dim, 1);
  const int active_procs = grid_dim * grid_dim;
  const int p = grid_dim;
  const int block_size = matrix_size_ / p;

  MPI_Comm active_comm{};
  int color = (world_rank < active_procs ? 0 : MPI_UNDEFINED);
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active_comm);

  bool is_active = (active_comm != MPI_COMM_NULL);
  MPI_Comm row_comm{};

  if (is_active) {
    int my_row = world_rank / p;
    int my_col = world_rank % p;

    std::vector<double> a_block(block_size * block_size);
    std::vector<double> b_block(block_size * block_size);
    std::vector<double> c_block(block_size * block_size, 0.0);

    if (world_rank == 0) {
      for (int proc = 0; proc < active_procs; proc++) {
        int row = proc / p;
        int col = proc % p;
        std::vector<double> tmp(block_size * block_size);
        for (int i = 0; i < block_size; i++) {
          int src = ((row * block_size + i) * matrix_size_) + (col * block_size);
          std::copy_n(&matrix_a_[src], block_size, &tmp[i * block_size]);
        }
        if (proc == 0) {
          a_block = tmp;
        } else {
          MPI_Send(tmp.data(), static_cast<int>(tmp.size()), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }
      }
      for (int proc = 0; proc < active_procs; proc++) {
        int row = proc / p;
        int col = proc % p;
        std::vector<double> tmp(block_size * block_size);
        for (int i = 0; i < block_size; i++) {
          int src = ((row * block_size + i) * matrix_size_) + (col * block_size);
          std::copy_n(&matrix_b_[src], block_size, &tmp[i * block_size]);
        }
        if (proc == 0) {
          b_block = tmp;
        } else {
          MPI_Send(tmp.data(), static_cast<int>(tmp.size()), MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
        }
      }
    } else {
      MPI_Recv(a_block.data(), static_cast<int>(a_block.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(b_block.data(), static_cast<int>(b_block.size()), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<double> orig_a_block = a_block;

    MPI_Comm_split(active_comm, my_row, my_col, &row_comm);

    for (int step = 0; step < p; step++) {
      a_block = orig_a_block;
      int root = (my_row + step) % p;
      MPI_Bcast(a_block.data(), static_cast<int>(a_block.size()), MPI_DOUBLE, root, row_comm);

#pragma omp parallel for
      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          double sum = 0;
          for (int k = 0; k < block_size; k++) {
            sum += a_block[(i * block_size) + k] * b_block[(k * block_size) + j];
          }
          c_block[(i * block_size) + j] += sum;
        }
      }

      int next = (my_row + 1) % p;
      int prev = (my_row - 1 + p) % p;
      MPI_Sendrecv_replace(b_block.data(), static_cast<int>(b_block.size()), MPI_DOUBLE, (prev * p) + my_col, 0,
                           (next * p) + my_col, 0, active_comm, MPI_STATUS_IGNORE);
    }

    if (world_rank == 0) {
      for (int i = 0; i < block_size; i++) {
        int dst = ((i + my_row * block_size) * matrix_size_) + (my_col * block_size);
        std::copy_n(&c_block[i * block_size], block_size, &matrix_c_[dst]);
      }
      for (int proc = 1; proc < active_procs; proc++) {
        std::vector<double> tmp(block_size * block_size);
        MPI_Recv(tmp.data(), static_cast<int>(tmp.size()), MPI_DOUBLE, proc, 0, active_comm, MPI_STATUS_IGNORE);
        int row = proc / p;
        int col = proc % p;
        for (int i = 0; i < block_size; i++) {
          int dst = ((i + row * block_size) * matrix_size_) + (col * block_size);
          std::copy_n(&tmp[i * block_size], block_size, &matrix_c_[dst]);
        }
      }
    } else {
      MPI_Send(c_block.data(), static_cast<int>(c_block.size()), MPI_DOUBLE, 0, 0, active_comm);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&active_comm);
  }

  MPI_Bcast(matrix_c_.data(), matrix_size_ * matrix_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(matrix_c_, out_ptr);
  return true;
}
