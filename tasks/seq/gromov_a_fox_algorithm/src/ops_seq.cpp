#include "seq/gromov_a_fox_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool gromov_a_fox_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }

  unsigned int matrix_size = input_size / 2;
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);

  A_ = std::vector<double>(in_ptr, in_ptr + matrix_size);
  B_ = std::vector<double>(in_ptr + matrix_size, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);

  n_ = static_cast<int>(std::sqrt(matrix_size));
  if (n_ * n_ != static_cast<int>(matrix_size)) {
    return false;
  }

  block_size_ = n_ / 2;
  return block_size_ > 0;
}

bool gromov_a_fox_algorithm_seq::TestTaskSequential::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));
  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

bool gromov_a_fox_algorithm_seq::TestTaskSequential::RunImpl() {
  int num_blocks = n_ / block_size_;

  for (int stage = 0; stage < num_blocks; ++stage) {
    for (int i = 0; i < n_; i += block_size_) {
      for (int j = 0; j < n_; j += block_size_) {
        for (int bi = i; bi < i + block_size_ && bi < n_; ++bi) {
          for (int bj = j; bj < j + block_size_ && bj < n_; ++bj) {
            for (int bk = stage * block_size_; bk < (stage + 1) * block_size_ && bk < n_; ++bk) {
              output_[(bi * n_) + bj] += A_[(bi * n_) + bk] * B_[(bk * n_) + bj];
            }
          }
        }
      }
    }
  }
  return true;
}

bool gromov_a_fox_algorithm_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}