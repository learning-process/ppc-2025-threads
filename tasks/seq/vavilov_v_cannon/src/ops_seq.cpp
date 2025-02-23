#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

bool vavilov_v_cannon_seq::CannonSequential::PreProcessingImpl() {
  N = static_cast<unsigned int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks = static_cast<unsigned int>(std::sqrt(N));
  block_size = N / num_blocks;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + N * N);
  B_.assign(b, b + N * N);
  C_.assign(N * N, 0);

  InitialShift();
  return true;
}

void vavilov_v_cannon_seq::CannonSequential::InitialShift() {
  std::vector<double> A_tmp = A_;
  std::vector<double> B_tmp = B_;

  for (unsigned int bi = 0; bi < num_blocks; ++bi) {
    for (unsigned int bj = 0; bj < num_blocks; ++bj) {
      unsigned int src_row = (bi + bj) % num_blocks;
      unsigned int src_col = (bj + bi) % num_blocks;
      for (unsigned int i = 0; i < block_size; ++i) {
        for (unsigned int j = 0; j < block_size; ++j) {
          B_[(bi * block_size + i) * N + (bj * block_size + j)] =
              B_tmp[(src_row * block_size + i) * N + (bj * block_size + j)];
          A_[(bi * block_size + i) * N + (bj * block_size + j)] =
              A_tmp[(bi * block_size + i) * N + (src_col * block_size + j)];
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
  for (unsigned int bi = 0; bi < N; bi += block_size) {
    for (unsigned int bj = 0; bj < N; bj += block_size) {
      for (unsigned int i = bi; i < bi + block_size; i++) {
        for (unsigned int j = bj; j < bj + block_size; j++) {
          double temp = 0.0;
          for (unsigned int k = 0; k < block_size; k++) {
            unsigned int row_A = bi + (i - bi);  // row_A index
            unsigned int col_A = bj + k;         // col_A index
            unsigned int row_B = bi + k;         // row_B index
            unsigned int col_B = bj + (j - bj);  // col_B index

            temp += A_[row_A * N + col_A] * B_[row_B * N + col_B];
          }

          C_[i * N + j] += temp;
        }
      }
    }
  }
}

void vavilov_v_cannon_seq::CannonSequential::ShiftBlocks() {
  std::vector<double> A_tmp = A_;
  std::vector<double> B_tmp = B_;

  for (unsigned int bi = 0; bi < num_blocks; ++bi) {
    for (unsigned int bj = 0; bj < num_blocks; ++bj) {
      unsigned int src_row = (bi + 1) % num_blocks;
      unsigned int src_col = (bj + 1) % num_blocks;
      for (unsigned int i = 0; i < block_size; ++i) {
        for (unsigned int j = 0; j < block_size; ++j) {
          B_[(bi * block_size + i) * N + (bj * block_size + j)] =
              B_tmp[(src_row * block_size + i) * N + (bj * block_size + j)];
          A_[(bi * block_size + i) * N + (bj * block_size + j)] =
              A_tmp[(bi * block_size + i) * N + (src_col * block_size + j)];
        }
      }
    }
  }
}

bool vavilov_v_cannon_seq::CannonSequential::RunImpl() {
  for (unsigned int iter = 0; iter < num_blocks; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }
  return true;
}

bool vavilov_v_cannon_seq::CannonSequential::PostProcessingImpl() {
  std::copy(C_.begin(), C_.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
