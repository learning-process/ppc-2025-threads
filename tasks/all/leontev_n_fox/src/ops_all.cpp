#include "all/leontev_n_fox/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include <mpi.h>
#include "core/util/include/util.hpp"

namespace leontev_n_fox_all {

double FoxALL::AtA(size_t i, size_t j) const {
  if (i >= n_ || j >= n_) {
    return 0.0;
  }
  return input_a_[(i * n_) + j];
}

double FoxALL::AtB(size_t i, size_t j) const {
  if (i >= n_ || j >= n_) {
    return 0.0;
  }
  return input_b_[(i * n_) + j];
}

std::vector<double> MatMul(std::vector<double>& a, std::vector<double>& b, size_t n) {
  std::vector<double> res(n * n, 0.0);
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      for (size_t l = 0; l < n; l++) {
        res[(i * n) + j] += a[(i * n) + l] * b[(l * n) + j];
      }
    }
  }
  return res;
}

void FoxALL::MatMulBlocks(size_t a_pos_x, size_t a_pos_y, size_t b_pos_x, size_t b_pos_y, size_t c_pos_x,
                          size_t c_pos_y, size_t size) {
  size_t row_max = (n_ >= c_pos_y) ? (n_ - c_pos_y) : 0;
  size_t col_max = (n_ >= c_pos_x) ? (n_ - c_pos_x) : 0;
  for (size_t j = 0; j < std::min(size, col_max); j++) {
    for (size_t i = 0; i < std::min(size, row_max); i++) {
      for (size_t l = 0; l < size; l++) {
        output_[((i + c_pos_y) * n_) + (j + c_pos_x)] += AtA(i + a_pos_y, l + a_pos_x) * AtB(l + b_pos_y, j + b_pos_x);
      }
    }
  }
}

bool FoxALL::PreProcessingImpl() {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    MPI_Init(NULL, NULL);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  // fix that
  grid_size_mpi_ = static_cast<size_t>(std::sqrt(size_));
  if (grid_size_mpi_ * grid_size_mpi_ != size_) {
    return false;
  }
  if (rank_ == 0) {
    size_t input_count = task_data->inputs_count[0];
    auto* double_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    n_ = reinterpret_cast<size_t*>(task_data->inputs[1])[0];
    input_a_.assign(double_ptr, double_ptr + (input_count / 2));
    input_b_.assign(double_ptr + (input_count / 2), double_ptr + input_count);
    output_.resize(task_data->outputs_count[0], 0.0);
  }
  MPI_Bcast(&n_, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (n_ % grid_size_mpi_ != 0) {
    return false;
  }
  block_size_mpi_ = n_ / grid_size_mpi_;
  local_a_.resize(block_size_mpi_ * block_size_mpi_);
  local_b_.resize(block_size_mpi_ * block_size_mpi_);
  local_c_.resize(block_size_mpi_ * block_size_mpi_, 0.0);
  if (rank_ == 0) {
    std::vector<double> send_buf(block_size_mpi_ * block_size_mpi_);
    for (size_t i = 0; i < grid_size_mpi_; i++) {
      for (size_t j = 0; j < grid_size_mpi_; j++) {
        for (size_t bi = 0; bi < block_size_mpi_; bi++) {
          for (size_t bj = 0; bj < block_size_mpi_; bj++) {
            send_buf[bi * block_size_mpi_ + bj] =
                input_a_[(i * block_size_mpi_ + bi) * n_ + (j * block_size_mpi_ + bj)];
          }
        }
        int dest = i * grid_size_mpi_ + j;
        if (dest != 0) {
          MPI_Send(send_buf.data(), send_buf.size(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        } else {
          local_a_ = send_buf;
        }
        for (size_t bi = 0; bi < block_size_mpi_; bi++) {
          for (size_t bj = 0; bj < block_size_mpi_; bj++) {
            send_buf[bi * block_size_mpi_ + bj] =
                input_b_[(i * block_size_mpi_ + bi) * n_ + (j * block_size_mpi_ + bj)];
          }
        }
        if (dest != 0) {
          MPI_Send(send_buf.data(), send_buf.size(), MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
        } else {
          local_b_ = send_buf;
        }
      }
    }
  } else {
    MPI_Status status;
    MPI_Recv(local_a_.data(), local_a_.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(local_b_.data(), local_b_.size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
  }
  return true;
}

bool FoxALL::ValidationImpl() { return (input_a_.size() == n_ * n_ && output_.size() == n_ * n_); }

bool FoxALL::RunImpl() {
  size_t row = rank_ / grid_size_mpi_;
  size_t col = rank_ % grid_size_mpi_;
  MPI_Comm row_comm, col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

  for (size_t step = 0; step < grid_size_mpi_; step++) {
    size_t root_col = (col + step) % grid_size_mpi_;
    std::vector<double> temp_a(block_size_mpi_ * block_size_mpi_);
    if (col == root_col) temp_a = local_a_;
    MPI_Bcast(temp_a.data(), temp_a.size(), MPI_DOUBLE, root_col, row_comm);
    size_t num_threads = ppc::util::GetPPCNumThreads();
    size_t q_thread = std::min(block_size_mpi_, static_cast<size_t>(std::sqrt(num_threads)));
    if (q_thread == 0) return false;
    size_t k_thread = (block_size_mpi_ + q_thread - 1) / q_thread;

    std::vector<std::thread> threads;
    for (size_t i_thread = 0; i_thread < q_thread; i_thread++) {
      for (size_t j_thread = 0; j_thread < q_thread; j_thread++) {
        threads.emplace_back([&, i_thread, j_thread, k_thread]() {
          size_t start_i = i_thread * k_thread;
          size_t end_i = std::min(start_i + k_thread, block_size_mpi_);
          size_t start_j = j_thread * k_thread;
          size_t end_j = std::min(start_j + k_thread, block_size_mpi_);
          for (size_t i = start_i; i < end_i; i++) {
            for (size_t j = start_j; j < end_j; j++) {
              double sum = 0.0;
              for (size_t l = 0; l < block_size_mpi_; l++) {
                sum += temp_a[i * block_size_mpi_ + l] * local_b_[l * block_size_mpi_ + j];
              }
              local_c_[i * block_size_mpi_ + j] += sum;
            }
          }
        });
      }
    }
    for (auto& t : threads) t.join();
    MPI_Sendrecv_replace(local_b_.data(), local_b_.size(), MPI_DOUBLE, (row + grid_size_mpi_ - 1) % grid_size_mpi_, 0,
                         (row + 1) % grid_size_mpi_, 0, col_comm, MPI_STATUS_IGNORE);
  }


  if (rank_ == 0) {
    for (size_t i = 0; i < block_size_mpi_; i++) {
      for (size_t j = 0; j < block_size_mpi_; j++) {
        output_[(row * block_size_mpi_ + i) * n_ + (col * block_size_mpi_ + j)] = local_c_[i * block_size_mpi_ + j];
      }
    }
    for (int r = 1; r < size_; r++) {
      std::vector<double> recv_buf(block_size_mpi_ * block_size_mpi_);
      MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      size_t r_row = r / grid_size_mpi_;
      size_t r_col = r % grid_size_mpi_;
      for (size_t i = 0; i < block_size_mpi_; i++) {
        for (size_t j = 0; j < block_size_mpi_; j++) {
          output_[(r_row * block_size_mpi_ + i) * n_ + (r_col * block_size_mpi_ + j)] =
              recv_buf[i * block_size_mpi_ + j];
        }
      }
    }
  } else {
    MPI_Send(local_c_.data(), local_c_.size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  return true;
}

bool FoxALL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace leontev_n_fox_all
