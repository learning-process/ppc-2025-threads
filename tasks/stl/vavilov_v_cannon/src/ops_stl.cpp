#include "stl/vavilov_v_cannon/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

bool vavilov_v_cannon_stl::CannonSTL::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(std::sqrt(N_));
  block_size_ = N_ / num_blocks_;

  auto *a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *b = reinterpret_cast<double *>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_stl::CannonSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

void vavilov_v_cannon_stl::CannonSTL::InitialShift() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  std::vector<std::thread> threads;

  auto shift_work = [&](int bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int src_row = (bi + bj) % num_blocks_;
      int src_col = (bj + bi) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          B_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              b_tmp[(src_row * block_size_ + i) * N_ + (bj * block_size_ + j)];
          A_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              a_tmp[(bi * block_size_ + i) * N_ + (src_col * block_size_ + j)];
        }
      }
    }
  };

  for (int bi = 0; bi < num_blocks_; ++bi) {
    threads.emplace_back(shift_work, bi);
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void vavilov_v_cannon_stl::CannonSTL::BlockMultiply() {
  std::vector<std::thread> threads;

  auto multiply_work = [&](int bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int bk = (bi + bj) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          double temp = 0.0;
          for (int k = 0; k < block_size_; ++k) {
            temp += A_[(bi * block_size_ + i) * N_ + (bk * block_size_ + k)] *
                    B_[(bk * block_size_ + k) * N_ + (bj * block_size_ + j)];
          }
          C_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] += temp;
        }
      }
    }
  };

  for (int bi = 0; bi < num_blocks_; ++bi) {
    threads.emplace_back(multiply_work, bi);
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void vavilov_v_cannon_stl::CannonSTL::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  std::vector<std::thread> threads;

  auto shift_work = [&](int bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int src_row = (bi + 1) % num_blocks_;
      int src_col = (bj + 1) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          B_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              b_tmp[(src_row * block_size_ + i) * N_ + (bj * block_size_ + j)];
          A_[(bi * block_size_ + i) * N_ + (bj * block_size_ + j)] =
              a_tmp[(bi * block_size_ + i) * N_ + (src_col * block_size_ + j)];
        }
      }
    }
  };

  for (int bi = 0; bi < num_blocks_; ++bi) {
    threads.emplace_back(shift_work, bi);
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

bool vavilov_v_cannon_stl::CannonSTL::RunImpl() {
  InitialShift();
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }
  return true;
}

bool vavilov_v_cannon_stl::CannonSTL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
