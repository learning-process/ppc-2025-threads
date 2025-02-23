#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

bool vavilov_v_cannon_seq::CannonSequential::PreProcessingImpl() {
  // Matrix with dim = (N * N). P - the total number of blocks. Each block have dim = (N/sqrt(P) * N/sqrt(P))
  // Without limitation of generality P = N
  N_ = static_cast<unsigned int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<unsigned int>(std::sqrt(N_));  // num_blocks in row/col. Not the total number of blocks
  block_size_ = N_ / num_blocks_;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + N_ * N_);
  B_.assign(b, b + N_ * N_);
  C_.assign(N_ * N_, 0);

  InitialShift();
  return true;
}

void vavilov_v_cannon_seq::CannonSequential::InitialShift() {
  std::vector<double> A_tmp = A_;
  std::vector<double> B_tmp = B_;

  for (unsigned int bi = 0; bi < num_blocks_; ++bi) {
    for (unsigned int bj = 0; bj < num_blocks_; ++bj) {
      unsigned int src_row = (bi + bj) % num_blocks_;
      unsigned int src_col = (bj + bi) % num_blocks_;
      for (unsigned int i = 0; i < block_size_; ++i) {
        for (unsigned int j = 0; j < block_size_; ++j) {
          B_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              B_tmp[(src_row * block_size_ + i) * N_ + (bj * block_size_ + j)];
          A_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              A_tmp[(bi * block_size_ + i) * N_ + (src_col * block_size_ + j)];
        }
      }
    }
  }
}

bool vavilov_v_cannon_seq::CannonSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

void vavilov_v_cannon_seq::CannonSequential::BlockMultiply() {
  for (unsigned int bi = 0; bi < N_; bi += block_size_) {
    for (unsigned int bj = 0; bj < N_; bj += block_size_) {
      for (unsigned int i = bi; i < bi + block_size_; i++) {
        for (unsigned int j = bj; j < bj + block_size_; j++) {
          double temp = 0.0;
          for (unsigned int k = 0; k < block_size_; k++) {
            unsigned int row_A = bi + (i - bi);  // row_A index
            unsigned int col_A = bj + k;         // col_A index
            unsigned int row_B = bi + k;         // row_B index
            unsigned int col_B = bj + (j - bj);  // col_B index

            temp += A_[row_A * N_ + col_A] * B_[row_B * N_ + col_B];
          }

          C_[i * N_ + j] += temp;
        }
      }
    }
  }
}

void vavilov_v_cannon_seq::CannonSequential::ShiftBlocks() {
  std::vector<double> A_tmp = A_;
  std::vector<double> B_tmp = B_;

  for (unsigned int bi = 0; bi < num_blocks_; ++bi) {
    for (unsigned int bj = 0; bj < num_blocks_; ++bj) {
      unsigned int src_row = (bi + 1) % num_blocks_;
      unsigned int src_col = (bj + 1) % num_blocks_;
      for (unsigned int i = 0; i < block_size_; ++i) {
        for (unsigned int j = 0; j < block_size_; ++j) {
          B_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              B_tmp[(src_row * block_size_ + i) * N_ + (bj * block_size_ + j)];
          A_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              A_tmp[(bi * block_size_ + i) * N_ + (src_col * block_size_ + j)];
        }
      }
    }
  }
}

bool vavilov_v_cannon_seq::CannonSequential::RunImpl() {
  for (unsigned int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }
  return true;
}

bool vavilov_v_cannon_seq::CannonSequential::PostProcessingImpl() {
  std::copy(C_.begin(), C_.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
