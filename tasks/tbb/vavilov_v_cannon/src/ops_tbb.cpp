#include "tbb/vavilov_v_cannon/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <vector>

#include "core/util/include/util.hpp"
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
/*

void vavilov_v_cannon_tbb::CannonTBB::InitialShift() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_), [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
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
    }
  });
}

void vavilov_v_cannon_tbb::CannonTBB::BlockMultiply() {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_),
      [&](const oneapi::tbb::blocked_range2d<int>& r) {
        std::vector<double> a_block(block_size_ * block_size_);
        std::vector<double> b_block(block_size_ * block_size_);

        for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
          for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
            int base_row = bi * block_size_;
            int base_col = bj * block_size_;

            for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
              for (int k = 0; k < block_size_ && base_col + k < N_; ++k) {
                a_block[i * block_size_ + k] = A_[(base_row + i) * N_ + (base_col + k)];
                b_block[k * block_size_ + i] = B_[(base_row + k) * N_ + (base_col + i)];
              }
            }

            for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
              int row = base_row + i;
              for (int j = 0; j < block_size_ && base_col + j < N_; ++j) {
                int col = base_col + j;
                double temp = 0.0;
                int k = 0;

                for (; k <= block_size_ - 4; k += 4) {
                  temp += a_block[i * block_size_ + k] * b_block[k * block_size_ + j] +
                          a_block[i * block_size_ + k + 1] * b_block[(k + 1) * block_size_ + j] +
                          a_block[i * block_size_ + k + 2] * b_block[(k + 2) * block_size_ + j] +
                          a_block[i * block_size_ + k + 3] * b_block[(k + 3) * block_size_ + j];
                }

                for (; k < block_size_ && base_row + k < N_; ++k) {
                  temp += a_block[i * block_size_ + k] * b_block[k * block_size_ + j];
                }

                C_[row * N_ + col] += temp;
              }
            }
          }
        }
      },
      oneapi::tbb::auto_partitioner());
}

void vavilov_v_cannon_tbb::CannonTBB::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_), [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
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
    }
  });
}

bool vavilov_v_cannon_tbb::CannonTBB::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&]() {
    InitialShift();
    for (int iter = 0; iter < num_blocks_; ++iter) {
      BlockMultiply();
      ShiftBlocks();
    }
  });
  return true;
}
*/
/*
void vavilov_v_cannon_tbb::CannonTBB::InitialShift() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_), [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
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
    }
  });
}
*/
/*
void vavilov_v_cannon_tbb::CannonTBB::BlockMultiply() {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_),
      [&](const oneapi::tbb::blocked_range2d<int>& r) {
        std::vector<double> a_block(block_size_ * block_size_);
        std::vector<double> b_block_trans(block_size_ * block_size_);

        for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
          for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
            int base_row = bi * block_size_;
            int base_col = bj * block_size_;

            // Копируем блоки в локальные буферы (b_block транспонируется)
            for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
              for (int j = 0; j < block_size_ && base_col + j < N_; ++j) {
                a_block[i * block_size_ + j] = A_[(base_row + i) * N_ + (base_col + j)];
                b_block_trans[j * block_size_ + i] = B_[(base_row + j) * N_ + (base_col + i)];
              }
            }

            // Умножение блоков через transform_reduce
            for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
              int row = base_row + i;
              for (int j = 0; j < block_size_ && base_col + j < N_; ++j) {
                int col = base_col + j;

                auto a_it = a_block.begin() + i * block_size_;
                auto b_it = b_block_trans.begin() + j * block_size_;

                double sum = tbb::parallel_reduce(
                    tbb::blocked_range<int>(0, block_size_), 0.0,
                    [&](const tbb::blocked_range<int>& r, double init) {
                      double local_sum = init;
                      for (int k = r.begin(); k != r.end(); ++k) {
                        local_sum += a_block[i * block_size_ + k] * b_block_trans[k * block_size_ + j];
                      }
                      return local_sum;
                    },
                    std::plus<>());

                C_[row * N_ + col] += sum;
              }
            }
          }
        }
      },
      oneapi::tbb::auto_partitioner());
}
*/
/*
void vavilov_v_cannon_tbb::CannonTBB::BlockMultiply() {
  std::vector<std::mutex> lock_C_(N_ * N_);
  int num_threads = oneapi::tbb::this_task_arena::max_concurrency();
  std::vector<std::vector<double>> local_Cs(num_threads, std::vector<double>(N_ * N_, 0.0));

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_),
                            [&](const oneapi::tbb::blocked_range2d<int>& r) {
                              int tid = oneapi::tbb::this_task_arena::current_thread_index();
                              auto& local_C = local_Cs[tid];

                              std::vector<double> a_block(block_size_ * block_size_);
                              std::vector<double> b_block_trans(block_size_ * block_size_);

                              for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
                                for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
                                  int base_row = bi * block_size_;
                                  int base_col = bj * block_size_;

                                  // Копируем блоки
                                  for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
                                    for (int j = 0; j < block_size_ && base_col + j < N_; ++j) {
                                      a_block[i * block_size_ + j] = A_[(base_row + i) * N_ + (base_col + j)];
                                      b_block_trans[j * block_size_ + i] = B_[(base_row + j) * N_ + (base_col + i)];
                                    }
                                  }

                                  // Блочное умножение
                                  for (int i = 0; i < block_size_ && base_row + i < N_; ++i) {
                                    for (int j = 0; j < block_size_ && base_col + j < N_; ++j) {
                                      double sum = 0.0;
                                      for (int k = 0; k < block_size_; ++k) {
                                        sum += a_block[i * block_size_ + k] * b_block_trans[j * block_size_ + k];
                                      }
                                      local_C[(base_row + i) * N_ + (base_col + j)] += sum;
                                    }
                                  }
                                }
                              }
                            });

  // Слияние результатов из всех потоков в C_
  for (int tid = 0; tid < num_threads; ++tid) {
    for (int i = 0; i < N_ * N_; ++i) {
      if (local_Cs[tid][i] != 0.0) {
        std::lock_guard<std::mutex> guard(lock_C_[i]);
        C_[i] += local_Cs[tid][i];
      }
    }
  }
}
*/
/*
void vavilov_v_cannon_tbb::CannonTBB::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;
  tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_), [&](const tbb::blocked_range2d<int>& r) {
    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
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
    }
  });
}
*/
/*
bool vavilov_v_cannon_tbb::CannonTBB::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&]() {
    InitialShift();
    for (int iter = 0; iter < num_blocks_; ++iter) {
      BlockMultiply();
      ShiftBlocks();
    }
  });
  return true;
}
*/

void vavilov_v_cannon_tbb::CannonTBB::BlockMultiply() {
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, num_blocks_, 0, num_blocks_),
                            [&](const oneapi::tbb::blocked_range2d<int>& r) {
                              for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
                                for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
                                  int base_row = bi * block_size_;
                                  int base_col = bj * block_size_;

                                  // Обнуляем блок результата
                                  for (int i = 0; i < block_size_; ++i)
                                    for (int j = 0; j < block_size_; ++j)
                                      if (base_row + i < N_ && base_col + j < N_)
                                        C_[(base_row + i) * N_ + (base_col + j)] = 0.0;

                                  // Сдвиг блоков по Каннону
                                  for (int bk = 0; bk < num_blocks_; ++bk) {
                                    int a_col = ((bi + bk) % num_blocks_) * block_size_;
                                    int b_row = ((bj + bk) % num_blocks_) * block_size_;

                                    // Прямое перемножение блоков
                                    for (int i = 0; i < block_size_; ++i) {
                                      if (base_row + i >= N_) continue;
                                      for (int j = 0; j < block_size_; ++j) {
                                        if (base_col + j >= N_) continue;
                                        double sum = 0.0;
                                        for (int k = 0; k < block_size_; ++k) {
                                          if (a_col + k >= N_ || b_row + k >= N_) continue;
                                          sum += A_[(base_row + i) * N_ + (a_col + k)] *
                                                 B_[(b_row + k) * N_ + (base_col + j)];
                                        }
                                        C_[(base_row + i) * N_ + (base_col + j)] += sum;
                                      }
                                    }
                                  }
                                }
                              }
                            });
}

bool vavilov_v_cannon_tbb::CannonTBB::RunImpl() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&]() { BlockMultiply(); });
  return true;
}

bool vavilov_v_cannon_tbb::CannonTBB::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
