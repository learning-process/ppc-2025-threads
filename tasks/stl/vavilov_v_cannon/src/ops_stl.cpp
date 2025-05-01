#include "stl/vavilov_v_cannon/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

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

  auto shift_work = [&](int bi_start, int bi_end) {
    for (int bi = bi_start; bi < bi_end; ++bi) {
      for (int bj = 0; bj < num_blocks_; ++bj) {
        int src_row = (bi + bj) % num_blocks_;
        int src_col = (bj + bi) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
            A_[((bi * block_size_ + i) * N_) + ((bj * block_size_) + j)] =
                a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
          }
        }
      }
    }
  };

  int blocks_per_thread = (num_blocks_ + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    int start = t * blocks_per_thread;
    int end = std::min(start + blocks_per_thread, num_blocks_);
    if (start < end) {
      threads.emplace_back(shift_work, start, end);
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void vavilov_v_cannon_stl::CannonSTL::BlockMultiply() {
  std::vector<std::thread> threads;
  std::vector<std::vector<double>> local_C;
  int num_threads_ = num_blocks_;

  auto multiply_work = [&](int bi_start, int bi_end, int thread_id) {
    std::vector<double> &local = local_C[thread_id];
    local.resize((bi_end - bi_start) * N_, 0.0);

    for (int bi = bi_start; bi < bi_end; bi += block_size_) {
      for (int bj = 0; bj < N_; bj += block_size_) {
        for (int i = bi; i < bi + block_size_; i++) {
          for (int j = bj; j < bj + block_size_; j++) {
            double temp = 0.0;
            for (int k = 0; k < block_size_; k++) {
              int row_a = bi + (i - bi);
              int col_a = bj + k;
              int row_b = bi + k;
              int col_b = bj + (j - bj);
              temp += A_[(row_a * N_) + col_a] * B_[(row_b * N_) + col_b];
            }
            local[((i - bi_start) * N_) + j] += temp;
          }
        }
      }
    }
  };

  local_C.resize(num_threads_);
  int blocks_per_thread = (num_blocks_ + num_threads_ - 1) / num_threads_;
  int bi_range = blocks_per_thread * block_size_;

  for (int t = 0; t < num_threads_; ++t) {
    int bi_start = t * bi_range;
    int bi_end = std::min(bi_start + bi_range, N_);
    if (bi_start < N_) {
      threads.emplace_back(multiply_work, bi_start, bi_end, t);
    }
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (int t = 0; t < num_threads_; ++t) {
    int bi_start = t * bi_range;
    int bi_end = std::min(bi_start + bi_range, N_);
    if (bi_start < N_) {
      const std::vector<double> &local = local_C[t];
      for (int i = bi_start; i < bi_end; ++i) {
        for (int j = 0; j < N_; ++j) {
          C_[(i * N_) + j] += local[((i - bi_start) * N_) + j];
        }
      }
    }
  }
}

void vavilov_v_cannon_stl::CannonSTL::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  std::vector<std::thread> threads;

  auto shift_work = [&](int bi_start, int bi_end) {
    for (int bi = bi_start; bi < bi_end; ++bi) {
      for (int bj = 0; bj < num_blocks_; ++bj) {
        int src_row = (bi + 1) % num_blocks_;
        int src_col = (bj + 1) % num_blocks_;
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
            A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
                a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
          }
        }
      }
    }
  };

  int blocks_per_thread = (num_blocks_ + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    int start = t * blocks_per_thread;
    int end = std::min(start + blocks_per_thread, num_blocks_);
    if (start < end) {
      threads.emplace_back(shift_work, start, end);
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

bool vavilov_v_cannon_stl::CannonSTL::RunImpl() {
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), num_blocks_);
  InitialShift(num_threads);
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply(num_threads);
    ShiftBlocks(num_threads);
  }
  return true;
}

bool vavilov_v_cannon_stl::CannonSTL::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
