#include "seq/morozov_e_lineare_image_filtering_block_gaussian/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  n_ = static_cast<int>(task_data->inputs_count[0]);
  m_ = static_cast<int>(task_data->inputs_count[0]);
  for (int i = 0; i < n_; ++i) {
    auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[i]);
    input_.emplace_back(std::vector<double>(in_ptr, in_ptr + m_));
  }

  int n_res = static_cast<int>(task_data->outputs_count[0]);
  int m_res = static_cast<int>(task_data->outputs_count[1]);
  res_ = std::vector<std::vector<double>>(n_res, std::vector<double>(m_res));

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
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < m_; ++j) {
      if (i == 0 || j == 0 || i == n_ - 1 || j == m_ - 1) {
        res_[i][j] = input_[i][j];
      } else {
        // std::cout<<i<<''<<j<<"\n";
        double sum = 0.0;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += input_[i + ki][j + kj] * kernel[ki + 1][kj + 1];
          }
        }
        res_[i][j] = sum;
      }
    }
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential::PostProcessingImpl() {
  for (int i = 0; i < n_; i++) {
    for (int j = 0; j < m_; j++) {
      reinterpret_cast<double *>(task_data->outputs[i])[j] = res_[i][j];
    }
  }
  return true;
}
