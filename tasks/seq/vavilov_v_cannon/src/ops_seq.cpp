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
          B_[(bi * block_size + i) * N + (bj * block_size + j)] = B_tmp[(src_row * block_size + i) * N + (bj * block_size + j)];
          A_[(bi * block_size + i) * N + (bj * block_size + j)] = A_tmp[(bi * block_size + i) * N + (src_col * block_size + j)];
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
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      for (unsigned int k = 0; k < N; ++k) {
        C_[i * N + j] += A_[i * N + k] * B_[k * N + j];
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
          B_[(bi * block_size + i) * N + (bj * block_size + j)] = B_tmp[(src_row * block_size + i) * N + (bj * block_size + j)];
          A_[(bi * block_size + i) * N + (bj * block_size + j)] = A_tmp[(bi * block_size + i) * N + (src_col * block_size + j)];
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
