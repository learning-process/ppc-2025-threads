#include "omp/konkov_i_sparse_matmul_ccs_omp/include/ops_omp.hpp"

#include "omp.h"

namespace konkov_i_sparse_matmul_ccs {

SparseMatmulTask::SparseMatmulTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}

bool SparseMatmulTask::ValidationImpl() {
  std::cout << "Проверка входных данных..." << std::endl;
  if (colsA != rowsB || rowsA <= 0 || colsB <= 0) {
    std::cout << "Ошибка: несовместимые размеры матриц!" << std::endl;
    return false;
  }
  if (A_col_ptr.empty() || B_col_ptr.empty()) {
    std::cout << "Ошибка: одна из матриц пустая!" << std::endl;
    return false;
  }
  return true;
}

bool SparseMatmulTask::PreProcessingImpl() {
  std::cout << "Подготовка структуры выходной матрицы..." << std::endl;
  C_col_ptr.resize(colsB + 1, 0);
  C_row_indices.clear();
  C_values.clear();
  return true;
}

bool SparseMatmulTask::RunImpl() {
  std::cout << "Запуск умножения матриц..." << std::endl;
  std::vector<std::unordered_map<int, double>> column_map(colsB);

#pragma omp parallel for
  for (int col_b = 0; col_b < colsB; ++col_b) {
    std::cout << "Обрабатываем столбец B: " << col_b << std::endl;
    for (int j = B_col_ptr[col_b]; j < B_col_ptr[col_b + 1]; ++j) {
      int row_b = B_row_indices[j];
      double val_b = B_values[j];

      if (row_b >= colsA) {
        continue;
      }

      for (int k = A_col_ptr[row_b]; k < A_col_ptr[row_b + 1]; ++k) {
        if (static_cast<size_t>(k) >= A_row_indices.size()) {
          continue;
        }

        int row_a = A_row_indices[k];
        double val_a = A_values[k];

#pragma omp critical
        column_map[col_b][row_a] += val_a * val_b;
      }
    }
  }

  C_col_ptr.resize(colsB + 1, 0);
  int count = 0;

  for (int col = 0; col < colsB; ++col) {
    std::vector<int> rows;
    for (const auto& pair : column_map[col]) {
      if (pair.second != 0) {
        rows.push_back(pair.first);
      }
    }
    std::sort(rows.begin(), rows.end());

    for (int row : rows) {
      C_values.push_back(column_map[col][row]);
      C_row_indices.push_back(row);
      count++;
    }
    C_col_ptr[col + 1] = count;
  }

  std::cout << "Умножение завершено!" << std::endl;
  return true;
}

bool SparseMatmulTask::PostProcessingImpl() {
  std::cout << "Финализация данных..." << std::endl;
  return true;
}

}  // namespace konkov_i_sparse_matmul_ccs
