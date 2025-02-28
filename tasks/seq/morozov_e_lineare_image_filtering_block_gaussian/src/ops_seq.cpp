#include "seq/morozov_e_lineare_image_filtering_block_gaussian/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <iostream>
typedef unsigned int uint;

bool morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  n = task_data->inputs_count[0];
  m = task_data->inputs_count[0];
  for (int i = 0; i < n; ++i) {
    double *in_ptr = reinterpret_cast<double *>(task_data->inputs[i]);
    input_.push_back(std::vector<double>(in_ptr, in_ptr + m));
  }

  uint nRes = task_data->outputs_count[0];
  uint mRes = task_data->outputs_count[1];
  res = std::vector<std::vector<double>>(nRes, std::vector<double>(mRes));

  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[1] == task_data->outputs_count[1] && task_data->inputs_count[1] > 0;
}

bool morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential::RunImpl() {
  // Ядро Гаусса 3x3
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
        res[i][j] = input_[i][j];
      } else {
        // std::cout<<i<<''<<j<<"\n";
        double sum = 0.0f;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += input_[i + ki][j + kj] * kernel[ki + 1][kj + 1];
          }
        }
        res[i][j] = sum;
      }
    }
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      reinterpret_cast<double *>(task_data->outputs[i])[j] = res[i][j];
    }
  }
  return true;
}
