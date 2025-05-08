#include "stl/gromov_a_fox_algorithm/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
void FoxBlockMul(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int n,
                 int block_size, int C_start_row_idx, int C_start_col_idx, int AB_common_dim_start_idx) {
  for (int bi = C_start_row_idx; bi < std::min(C_start_row_idx + block_size, n); ++bi) {
    for (int bj = C_start_col_idx; bj < std::min(C_start_col_idx + block_size, n); ++bj) {
      double sum = 0.0;
      for (int bk = AB_common_dim_start_idx; bk < std::min(AB_common_dim_start_idx + block_size, n); ++bk) {
        if ((bi * n + bk < a.size()) && (bk * n + bj < b.size())) {
          sum += a[(bi * n) + bk] * b[(bk * n) + bj];
        }
      }
      if (bi * n + bj < c.size()) {
        c[(bi * n) + bj] += sum;
      }
    }
  }
}

}  // namespace

namespace gromov_a_fox_algorithm_stl {

bool TestTaskSTL::PreProcessingImpl() {
  if (task_data->inputs_count.empty() || task_data->inputs.empty() || task_data->inputs_count[0] == 0) {
    if (!task_data->inputs_count.empty() && task_data->inputs_count[0] == 0) {
      if (task_data->outputs_count.empty() || task_data->outputs_count[0] != 0) {
        return false;
      }
      n_ = 0;
      block_size_ = 0;
      A_.clear();
      B_.clear();
      output_.clear();
      return true;
    }
    return false;
  }

  unsigned int total_input_elements = task_data->inputs_count[0];
  if (total_input_elements % 2 != 0) {
    return false;
  }

  unsigned int elements_per_matrix = total_input_elements / 2;

  if (task_data->outputs_count.empty() || task_data->outputs.empty() ||
      task_data->outputs_count[0] != elements_per_matrix) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  A_.assign(in_ptr, in_ptr + elements_per_matrix);
  B_.assign(in_ptr + elements_per_matrix, in_ptr + total_input_elements);
  output_.assign(elements_per_matrix, 0.0);

  n_ = static_cast<int>(std::sqrt(elements_per_matrix));
  if (n_ * n_ != static_cast<int>(elements_per_matrix)) {
    return false;
  }

  if (n_ == 0) {
    block_size_ = 0;
    return true;
  }

  int optimal_block_size_sqrt = static_cast<int>(std::sqrt(n_));
  if (optimal_block_size_sqrt == 0) optimal_block_size_sqrt = 1;

  block_size_ = 1;

  for (int i = optimal_block_size_sqrt; i >= 1; --i) {
    if (n_ % i == 0) {
      block_size_ = i;
      break;
    }
  }
  for (int i = optimal_block_size_sqrt + 1; i <= n_; ++i) {
    if (n_ % i == 0) {
      if (std::abs(i - optimal_block_size_sqrt) < std::abs(block_size_ - optimal_block_size_sqrt)) {
        block_size_ = i;
      }
      break;
    }
  }

  return block_size_ > 0;
}

bool TestTaskSTL::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty() || task_data->inputs.empty() ||
      task_data->outputs.empty()) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  if (input_size == 0) {
    return task_data->outputs_count[0] == 0;
  }

  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));

  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

void TestTaskSTL::WorkerFunction(int start_block_idx, int end_block_idx, int num_blocks_dim) {
  for (int linear_idx = start_block_idx; linear_idx < end_block_idx; ++linear_idx) {
    int block_row_C = linear_idx / num_blocks_dim;
    int block_col_C = linear_idx % num_blocks_dim;

    int C_start_row = block_row_C * block_size_;
    int C_start_col = block_col_C * block_size_;

    for (int step = 0; step < num_blocks_dim; ++step) {
      int k_shifted_block_idx = (block_row_C + step) % num_blocks_dim;
      FoxBlockMul(A_, B_, output_, n_, block_size_, C_start_row, C_start_col, k_shifted_block_idx * block_size_);
    }
  }
}

bool TestTaskSTL::RunImpl() {
  if (n_ == 0) {
    return true;
  }
  if (block_size_ == 0 && n_ > 0) {
    return false;
  }

  const int num_blocks_dim = (n_ + block_size_ - 1) / block_size_;
  const int total_C_blocks_to_compute = num_blocks_dim * num_blocks_dim;

  if (total_C_blocks_to_compute == 0 && n_ > 0) {
    return false;
  }
  if (total_C_blocks_to_compute == 0 && n_ == 0) {
    return true;
  }

  int num_threads_to_use = ppc::util::GetPPCNumThreads();
  if (num_threads_to_use <= 0) {
    num_threads_to_use = 1;
  }

  num_threads_to_use = std::min(num_threads_to_use, total_C_blocks_to_compute);
  if (num_threads_to_use == 0 && total_C_blocks_to_compute > 0) num_threads_to_use = 1;

  std::vector<std::thread> threads;
  threads.reserve(num_threads_to_use);

  int blocks_per_thread_base = total_C_blocks_to_compute / num_threads_to_use;
  int remaining_blocks = total_C_blocks_to_compute % num_threads_to_use;
  int current_block_start_idx = 0;

  for (int i = 0; i < num_threads_to_use; ++i) {
    int blocks_for_this_thread = blocks_per_thread_base + (i < remaining_blocks ? 1 : 0);
    if (blocks_for_this_thread == 0) continue;

    int current_block_end_idx = current_block_start_idx + blocks_for_this_thread;
    threads.emplace_back(&TestTaskSTL::WorkerFunction, this, current_block_start_idx, current_block_end_idx,
                         num_blocks_dim);
    current_block_start_idx = current_block_end_idx;
  }

  for (auto& th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  if (task_data->outputs_count.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (n_ == 0) {
    return task_data->outputs_count[0] == 0;
  }

  if (output_.size() != task_data->outputs_count[0]) {
    return false;
  }

  double* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}

}  // namespace gromov_a_fox_algorithm_stl