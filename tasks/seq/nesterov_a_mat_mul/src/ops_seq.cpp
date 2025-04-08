#include "seq/nesterov_a_mat_mul/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool nesterov_a_mat_mul_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  int sqrt_input_size = static_cast<int>(std::sqrt(input_size));
  if (sqrt_input_size * sqrt_input_size != input_size) {
    return false; // input_size is not a perfect square
  }
  rc_size_ = sqrt_input_size;
  return true;
}

bool nesterov_a_mat_mul_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool nesterov_a_mat_mul_seq::TestTaskSequential::RunImpl() {
  // Multiply matrices
  for (int i = 0; i < rc_size_; ++i) {
    for (int j = 0; j < rc_size_; ++j) {
      for (int k = 0; k < rc_size_; ++k) {
        output_[(i * rc_size_) + j] += input_[(i * rc_size_) + k] * input_[(k * rc_size_) + j];
      }
    }
  }
  return true;
}

bool nesterov_a_mat_mul_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
