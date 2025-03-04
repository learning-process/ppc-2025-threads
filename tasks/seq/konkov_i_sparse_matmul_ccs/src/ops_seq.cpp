#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

#include <iostream>

namespace konkov_i_sparse_matmul_ccs {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(task_data) {}

bool SparseMatmulTask::ValidationImpl() {
  // Проверка корректности входных данных
  if (A_values.size() == 0 || B_values.size() == 0) {
    std::cerr << "Error: Empty matrix data\n";
    return false;
  }
  if (colsA != rowsB) {
    std::cerr << "Error: Matrices dimensions mismatch\n";
    return false;
  }
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() {
  // Подготовка данных для умножения
  // Например, можем создать дополнительные структуры для столбцов и значений
  return true;
}

bool SparseMatmulTask::RunImpl() {
  // Реализация алгоритма умножения разреженных матриц
  C_values.resize(rowsA * colsB, 0.0);

  // Процесс умножения
  for (size_t i = 0; i < A_values.size(); ++i) {
    for (size_t j = 0; j < B_values.size(); ++j) {
      // Алгоритм умножения разреженных матриц в формате CCS
      C_values[i] += A_values[i] * B_values[j];  // Пример простого умножения
    }
  }
  return true;
}

bool SparseMatmulTask::PostProcessingImpl() {
  // Завершение работы задачи, обработка результата умножения
  std::cout << "Result matrix C:\n";
  for (const auto& value : C_values) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  return true;
}

}  // namespace seq
