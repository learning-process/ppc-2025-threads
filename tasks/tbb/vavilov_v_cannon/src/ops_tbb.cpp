#include "tbb/vavilov_v_cannon/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool vavilov_v_cannon_tbb::CannonTBB::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(std::sqrt(N_));
  block_size_ = N_ / num_blocks_;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_tbb::CannonTBB::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

void vavilov_v_cannon_tbb::CannonTBB::InitialShift() {
  tbb::parallel_for(tbb::blocked_range<int>(0, num_blocks_, 4), [&](const tbb::blocked_range<int>& r) {
    for (int bi = r.begin(); bi != r.end(); ++bi) {
      for (int bj = 0; bj < num_blocks_; ++bj) {
        int src_row = (bi + bj) % num_blocks_;
        int src_col = (bj + bi) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {

          std::swap_ranges(
              B_.begin() + (bi * block_size_ + i) * N_ + (bj * block_size_),
              B_.begin() + (bi * block_size_ + i) * N_ + (bj * block_size_ + block_size_),
              B_.begin() + (src_row * block_size_ + i) * N_ + (bj * block_size_)
          );

          for (int j = 0; j < block_size_; ++j) {
            std::swap(
                A_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)],
                A_[(bi * block_size_ + i) * N_ + (src_col * block_size_ + j)]
            );
          }
        }
      }
    }
  });
}

void vavilov_v_cannon_tbb::CannonTBB::BlockMultiply() {
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 4, 0, num_blocks_, 4),
                    [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
        int row_offset = bi * block_size_;
        int col_offset = bj * block_size_;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            double temp = 0.0;
            for (int k = 0; k < block_size_; ++k) {
              temp += A_[(row_offset + i) * N_ + (col_offset + k)] *
                      B_[(row_offset + k) * N_ + (col_offset + j)];
            }
            C_[(row_offset + i) * N_ + (col_offset + j)] += temp;
          }
        }
      }
    }
  });
}

void vavilov_v_cannon_tbb::CannonTBB::ShiftBlocks() {
  tbb::parallel_for(tbb::blocked_range<int>(0, num_blocks_, 4), [&](const tbb::blocked_range<int>& r) {
    for (int bi = r.begin(); bi != r.end(); ++bi) {
      for (int bj = 0; bj < num_blocks_; ++bj) {
        int src_row = (bi + 1) % num_blocks_;
        int src_col = (bj + 1) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {

          std::swap_ranges(
              B_.begin() + (bi * block_size_ + i) * N_ + (bj * block_size_),
              B_.begin() + (bi * block_size_ + i) * N_ + (bj * block_size_ + block_size_),
              B_.begin() + (src_row * block_size_ + i) * N_ + (bj * block_size_)
          );

          for (int j = 0; j < block_size_; ++j) {
            std::swap(
                A_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)],
                A_[(bi * block_size_ + i) * N_ + (src_col * block_size_ + j)]
            );
          }
        }
      }
    }
  });
}
bool vavilov_v_cannon_tbb::CannonTBB::RunImpl() {
  tbb::task_arena arena(tbb::task_arena::automatic);
  arena.execute([&]() {
    InitialShift();
    for (int iter = 0; iter < num_blocks_; ++iter) {
      BlockMultiply();
      ShiftBlocks();
    }
  });
  return true;
}

bool vavilov_v_cannon_tbb::CannonTBB::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
