#include "all/morozov_e_lineare_image_filtering_block_gaussian_all/include/ops_all.hpp"

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>
namespace morozov_e_lineare_image_filtering_block_gaussian_all {
std::pair<int, int> GetStartEndIndices(int countProc, int currRunkProc, int arraySize) {
  int start, end;
  int count = arraySize / countProc;
  int rem = arraySize % countProc;
  if (countProc < arraySize) {
    if (countProc % arraySize == 0) {
      start = currRunkProc * count;
      end = start + count;
    } else {
      if (currRunkProc < rem) {
        start = (currRunkProc) * (count + 1);
        end = start + count + 1;
      } else {
        start = rem * (count + 1) + (currRunkProc - rem) * (count);
        end = start + count;
      }
    }
  } else {
    if (currRunkProc < arraySize) {
      start = currRunkProc;
      end = start + 1;
    }
  }
  return {start, end};
}
}  // namespace morozov_e_lineare_image_filtering_block_gaussian_all
bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::PreProcessingImpl() {
  n_ = static_cast<int>(task_data->inputs_count[0]);
  m_ = static_cast<int>(task_data->inputs_count[1]);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (m_ * n_));
  res_ = std::vector<double>(n_ * m_, 0);
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0 &&
           task_data->inputs_count[1] == task_data->outputs_count[1] && task_data->inputs_count[1] > 0;
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::RunImpl() {
  // clang-format off
  const std::vector<std::vector<double>> kernel = {
      {1.0 / 16, 2.0 / 16, 1.0 / 16},
      {2.0 / 16, 4.0 / 16, 2.0 / 16},
      {1.0 / 16, 2.0 / 16, 1.0 / 16}};
  // clang-format on
  // Алгоритм вычисления диапазона вычисления для каждого процесса
  auto [start, end] =
      morozov_e_lineare_image_filtering_block_gaussian_all::GetStartEndIndices(world_.size(), world_.rank(), n_);

#pragma omp parallel for
  for (int i = start; i < end; ++i) {
    for (int j = 0; j < m_; ++j) {
      if (i == 0 || j == 0 || i == n_ - 1 || j == m_ - 1) {
        res_[(i * m_) + j] = input_[(i * m_) + j];
      } else {
        double sum = 0.0;
        // Применяем ядро к текущему пикселю и его соседям
        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            sum += input_[((i + ki) * m_) + (j + kj)] * kernel[ki + 1][kj + 1];
          }
        }
        res_[(i * m_) + j] = sum;
      }
    }
  }
  if (world_.rank() == 0) {
    // Процесс 0 собирает данные
    for (int p = 1; p < world_.size(); ++p) {
      int start_p = 0;
      int end_p = 0;
      // Получаем диапазон от процесса p
      world_.recv(p, 0, &start_p, 1);
      world_.recv(p, 0, &end_p, 1);
      // Получаем все данные разом
      std::vector<double> temp((end_p - start_p) * m_);
      world_.recv(p, 0, temp.data(), (end_p - start_p) * m_);
      // Копируем в res_
      for (int i = start_p; i < end_p; ++i) {
        for (int j = 0; j < m_; ++j) {
          res_[(i * m_) + j] = temp[((i - start_p) * m_) + j];
        }
      }
    }
  } else {
    // Отправляем диапазон и данные
    world_.send(0, 0, &start, 1);
    world_.send(0, 0, &end, 1);
    // Отправляем все данные разом
    std::vector<double> temp((end - start) * m_);
    for (int i = start; i < end; ++i) {
      for (int j = 0; j < m_; ++j) {
        temp[((i - start) * m_) + j] = res_[(i * m_) + j];
      }
    }
    world_.send(0, 0, temp.data(), (end - start) * m_);
  }
  return true;
}

bool morozov_e_lineare_image_filtering_block_gaussian_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (int i = 0; i < n_; i++) {
      for (int j = 0; j < m_; j++) {
        reinterpret_cast<double *>(task_data->outputs[0])[(i * m_) + j] = res_[(i * m_) + j];
      }
    }
  }
  return true;
}
