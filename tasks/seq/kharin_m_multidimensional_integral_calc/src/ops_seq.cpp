#include "seq/kharin_m_multidimensional_integral_calc/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::PreProcessingImpl() {
  // Чтение входных данных из task_data.
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  // Результат пока равен 0.
  output_result_ = 0;

  // Определяем размер квадратной сетки: grid_size = sqrt(input_size).
  grid_size_ = static_cast<size_t>(std::sqrt(input_size));
  return true;
}

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::ValidationImpl() {
  // Проверяем, что число входных элементов является полным квадратом
  // и что размер выходного буфера равен 1 (результат интеграла).
  unsigned int input_size = task_data->inputs_count[0];
  size_t n = static_cast<size_t>(std::sqrt(input_size));
  return ((n * n == input_size) && (task_data->outputs_count[0] == 1));
}

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::RunImpl() {
  // Многошаговая схема: сначала интегрируем по строкам (вычисляем сумму значений в каждой строке),
  // затем суммируем полученные значения.
  int total = 0;
  for (size_t i = 0; i < grid_size_; ++i) {
    int row_sum = 0;
    for (size_t j = 0; j < grid_size_; ++j) {
      row_sum += input_[i * grid_size_ + j];
    }
    total += row_sum;
  }
  output_result_ = total;
  return true;
}

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::PostProcessingImpl() {
  // Записываем результат вычисления интеграла в выходной буфер.
  reinterpret_cast<int *>(task_data->outputs[0])[0] = output_result_;
  return true;
}