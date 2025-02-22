#include "seq/vavilov_v_cannon/include/ops_seq.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include "core/task/include/task.hpp"
#include "core/perf/include/perf.hpp"

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

  for (unsigned int i = 0; i < num_blocks; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      A_[i * block_size * N + j] = A_tmp[i * block_size * N + (j + i * block_size) % N];
    }
  }

  for (unsigned int j = 0; j < num_blocks; ++j) {
    for (unsigned int i = 0; i < N; ++i) {
      B_[i * N + j * block_size] = B_tmp[((i + j * block_size) % N) * N + j * block_size];
    }
  }
}

bool vavilov_v_cannon_seq::CannonSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

void vavilov_v_cannon_seq::CannonSequential::BlockMultiply() {
  for (unsigned int i = 0; i < num_blocks; ++i) {
    for (unsigned int j = 0; j < num_blocks; ++j) {
      for (unsigned int k = 0; k < num_blocks; ++k) {
        double* C_block = &C_[(i * block_size) * N + j * block_size];
        double* A_block = &A_[(i * block_size) * N + k * block_size];
        double* B_block = &B_[(k * block_size) * N + j * block_size];

        for (unsigned int bi = 0; bi < block_size; ++bi) {
          for (unsigned int bj = 0; bj < block_size; ++bj) {
            for (unsigned int bk = 0; bk < block_size; ++bk) {
              C_block[bi * N + bj] += A_block[bi * N + bk] * B_block[bk * N + bj];
            }
          }
        }
      }
    }
  }
}

void vavilov_v_cannon_seq::CannonSequential::ShiftBlocks() {
  std::vector<double> A_tmp = A_;
  std::vector<double> B_tmp = B_;

  for (unsigned int i = 0; i < num_blocks; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      A_[(i * block_size * N) + j] = A_tmp[(i * block_size * N) + (j + block_size) % N];
    }
  }

  for (unsigned int j = 0; j < num_blocks; ++j) {
    for (unsigned int i = 0; i < N; ++i) {
      B_[i * N + j * block_size] = B_tmp[((i + block_size) % N) * N + j * block_size];
    }
  }
}

bool vavilov_v_cannon_seq::CannonSequential::RunImpl() {
  for (unsigned int iter = 0; iter < block_size; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }
  return true;
}

bool vavilov_v_cannon_seq::CannonSequential::PostProcessingImpl() {
  std::copy(C_.begin(), C_.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
