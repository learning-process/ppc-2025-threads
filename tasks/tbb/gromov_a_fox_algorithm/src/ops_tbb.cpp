#include "tbb/gromov_a_fox_algorithm/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

namespace {
void FoxBlockMul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n,
                 int block_size, int stage, int i, int j) {
  int start_k = stage * block_size;
  for (int bi = i; bi < i + block_size && bi < n; ++bi) {
    for (int bj = j; bj < j + block_size && bj < n; ++bj) {
      for (int bk = start_k; bk < std::min((stage + 1) * block_size, n); ++bk) {
        C[(bi * n) + bj] += A[(bi * n) + bk] * B[(bk * n) + bj];
      }
    }
  }
}
}  // namespace

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }

  unsigned int matrix_size = input_size / 2;
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  A_ = std::vector<double>(in_ptr, in_ptr + matrix_size);
  B_ = std::vector<double>(in_ptr + matrix_size, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);

  n_ = static_cast<int>(std::sqrt(matrix_size));
  if (n_ * n_ != static_cast<int>(matrix_size)) {
    return false;
  }

  block_size_ = n_ / 2;
  for (int i = 1; i <= n_; ++i) {
    if (n_ % i == 0) {
      block_size_ = i;
      break;
    }
  }
  return block_size_ > 0;
}

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));
  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::RunImpl() {
  int num_blocks = (n_ + block_size_ - 1) / block_size_;

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    tbb::task_group tg;
    for (int stage = 0; stage < num_blocks; ++stage) {
      for (int i = 0; i < n_; i += block_size_) {
        for (int j = 0; j < n_; j += block_size_) {
          tg.run([&, i, j, stage] { FoxBlockMul(A_, B_, output_, n_, block_size_, stage, i, j); });
        }
      }
      tg.wait();
    }
  });
  return true;
}

bool gromov_a_fox_algorithm_tbb::TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}