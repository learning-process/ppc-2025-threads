#include "seq/moiseev_a_mult_mat/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool moiseev_a_mult_mat_seq::MultMatSequential::PreProcessingImpl() {

  unsigned int input_size_A = task_data->inputs_count[0];
  unsigned int input_size_B = task_data->inputs_count[1];
  
  auto* in_ptr_A = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_B = reinterpret_cast<double*>(task_data->inputs[1]);
  
  matrix_A_ = std::vector<double>(in_ptr_A, in_ptr_A + input_size_A);
  matrix_B_ = std::vector<double>(in_ptr_B, in_ptr_B + input_size_B);
  

  unsigned int output_size = task_data->outputs_count[0];
  matrix_C_ = std::vector<double>(output_size, 0.0);
  
  matrix_size_ = static_cast<int>(std::sqrt(input_size_A));
  
  block_size_ = static_cast<int>(std::sqrt(matrix_size_));
  if (matrix_size_ % block_size_ != 0) {
    block_size_ = 1;
  }
  
  num_blocks_ = matrix_size_ / block_size_;
  
  return true;
}

bool moiseev_a_mult_mat_seq::MultMatSequential::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool moiseev_a_mult_mat_seq::MultMatSequential::RunImpl() {

  for (int i_block = 0; i_block < num_blocks_; ++i_block) {
    for (int j_block = 0; j_block < num_blocks_; ++j_block) {
      for (int s = 0; s < num_blocks_; ++s) {
        int a_block_j = (i_block + s) % num_blocks_;
        int b_block_i = a_block_j;
  
        int i_start = i_block * block_size_;
        int j_start = j_block * block_size_;
        int a_j_start = a_block_j * block_size_;
        int b_i_start = b_block_i * block_size_;
  
        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < block_size_; ++k) {
              double a_val = matrix_A_[ (i_start + i) * matrix_size_ + (a_j_start + k) ];
              double b_val = matrix_B_[ (b_i_start + k) * matrix_size_ + (j_start + j) ];
              sum += a_val * b_val;
            }
            matrix_C_[ (i_start + i) * matrix_size_ + (j_start + j) ] += sum;
          }
        }
      }
    }
  }
  return true;
}

bool moiseev_a_mult_mat_seq::MultMatSequential::PostProcessingImpl() {
  for (size_t i = 0; i < matrix_C_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = matrix_C_[i];
  }
  return true;
}
