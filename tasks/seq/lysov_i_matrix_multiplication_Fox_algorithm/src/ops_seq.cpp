#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  N = reinterpret_cast<std::size_t *>(task_data->inputs[0])[0];
  block_size = reinterpret_cast<std::size_t *>(task_data->inputs[3])[0];
  A.resize(N * N);
  B.resize(N * N);
  C.resize(N * N, 0.0);
  std::copy(reinterpret_cast<double *>(task_data->inputs[1]), reinterpret_cast<double *>(task_data->inputs[1]) + N * N,
            A.begin());
  std::copy(reinterpret_cast<double *>(task_data->inputs[2]), reinterpret_cast<double *>(task_data->inputs[2]) + N * N,
            B.begin());
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::ValidationImpl() {
  N = reinterpret_cast<std::size_t *>(task_data->inputs[0])[0];
  block_size = reinterpret_cast<std::size_t *>(task_data->inputs[3])[0];
  if (task_data->inputs_count.size() != 3 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[1] != N * N || task_data->inputs_count[0] != N * N) {
    return false;
  }
  return task_data->outputs_count[0] == N * N && block_size > 0;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::RunImpl() {
  std::size_t numBlocks = (N + block_size - 1) / block_size;
  for (std::size_t step = 0; step < numBlocks; ++step) {
    for (std::size_t i = 0; i < numBlocks; ++i) {
      std::size_t aBlockRow = (i + step) % numBlocks;

      for (std::size_t j = 0; j < numBlocks; ++j) {
        std::size_t block_h = std::min(block_size, N - i * block_size);
        std::size_t block_w = std::min(block_size, N - j * block_size);

        for (std::size_t ii = 0; ii < block_h; ++ii) {
          for (std::size_t jj = 0; jj < block_w; ++jj) {
            double sum = 0.0;
            for (std::size_t kk = 0; kk < std::min(block_size, N - aBlockRow * block_size); ++kk) {
              std::size_t rowA = i * block_size + ii;
              std::size_t colA = aBlockRow * block_size + kk;
              std::size_t rowB = aBlockRow * block_size + kk;
              std::size_t colB = j * block_size + jj;

              if (rowA < N && colA < N && rowB < N && colB < N) {
                sum += A[rowA * N + colA] * B[rowB * N + colB];
              }
            }
            std::size_t rowC = i * block_size + ii;
            std::size_t colC = j * block_size + jj;
            if (rowC < N && colC < N) {
              C[rowC * N + colC] += sum;
            }
          }
        }
      }
    }
  }

  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(C.begin(), C.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
