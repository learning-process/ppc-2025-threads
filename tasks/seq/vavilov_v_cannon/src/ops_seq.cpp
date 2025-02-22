#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

bool vavilov_v_cannon_seq::CannonSequential::PreProcessingImpl() {
  N = static_cast<unsigned int>(std::sqrt(task_data->inputs_count[0] / 2));
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

  unsigned int blockSize = N / std::sqrt(N * N);

  for (unsigned int bi = 0; bi < gridSize; ++bi) {
    for (unsigned int bj = 0; bj < gridSize; ++bj) {
      unsigned int src_col = (bj + bi) % gridSize;
      for (unsigned int i = 0; i < blockSize; ++i) {
        for (unsigned int j = 0; j < blockSize; ++j) {
          A_[(bi * blockSize + i) * N + (bj * blockSize + j)] = A_tmp[(bi * blockSize + i) * N + (src_col * blockSize + j)];
        }
      }
    }
  }

  for (unsigned int bi = 0; bi < gridSize; ++bi) {
    for (unsigned int bj = 0; bj < gridSize; ++bj) {
      unsigned int src_row = (bi + bj) % gridSize;
      for (unsigned int i = 0; i < blockSize; ++i) {
        for (unsigned int j = 0; j < blockSize; ++j) {
          B_[(bi * blockSize + i) * N + (bj * blockSize + j)] = B_tmp[(src_row * blockSize + i) * N + (bj * blockSize + j)];
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

  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      A_[i * N + j] = A_tmp[i * N + (j + 1) % N];
      B_[i * N + j] = B_tmp[((i + 1) % N) * N + j];
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
