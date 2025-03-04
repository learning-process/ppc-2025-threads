#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_pipeline_run) {
  // Инициализация матриц
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_a(400, false, 400);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_b(400, false, 400);

  std::vector<Complex> input_data = {};
  std::vector<Complex> vector_a;
  std::vector<Complex> vector_b;
  std::vector<Complex> result_vector(sparse_matrix_a.columnCount * sparse_matrix_b.rowCount * 100, 0);

  // Заполнение матриц случайными значениями
  for (unsigned int row = 0; row < 150; row++) {
    for (unsigned int col = 0; col < 150; col++) {
      sparse_matrix_a.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)),
                                    static_cast<int>(col));
    }
  }
  for (unsigned int row = 50; row < 140; row++) {
    for (unsigned int col = 50; col < 150; col++) {
      sparse_matrix_b.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)),
                                    static_cast<int>(col));
    }
  }

  // Преобразование матриц в векторы
  vector_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_a);
  vector_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_b);

  // Подготовка входных данных для задачи
  input_data.reserve(vector_a.size() + vector_b.size());
  input_data.insert(input_data.end(), vector_a.begin(), vector_a.end());
  input_data.insert(input_data.end(), vector_b.begin(), vector_b.end());

  // Инициализация структуры данных для задачи
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_vector.data()));
  task_data->outputs_count.emplace_back(result_vector.size());

  // Создание задачи
  auto sequential_task =
      std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(task_data);

  // Настройка атрибутов производительности
  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Инициализация результатов производительности
  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(sequential_task);
  performance_analyzer->PipelineRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);

  // Преобразование результата в матрицу
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS final_result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(result_vector);

  // Умножение матриц вручную для сравнения
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS expected_result(sparse_matrix_a.rowCount, false, sparse_matrix_b.columnCount);
  for (int i = 0; i < sparse_matrix_a.rowCount; ++i) {
    for (int j = 0; j < sparse_matrix_b.columnCount; ++j) {
      Complex sum(0, 0);
      for (int k = 0; k < sparse_matrix_a.columnCount; ++k) {
        // Поиск элементов в матрицах для умножения
        Complex a_value = 0;
        Complex b_value = 0;
        for (int idx_a = sparse_matrix_a.rowPointers[i]; idx_a < sparse_matrix_a.rowPointers[i + 1]; ++idx_a) {
          if (sparse_matrix_a.columnIndices[idx_a] == k) {
            a_value = sparse_matrix_a.data[idx_a];
            break;
          }
        }
        for (int idx_b = sparse_matrix_b.rowPointers[k]; idx_b < sparse_matrix_b.rowPointers[k + 1]; ++idx_b) {
          if (sparse_matrix_b.columnIndices[idx_b] == j) {
            b_value = sparse_matrix_b.data[idx_b];
            break;
          }
        }
        sum += a_value * b_value;
      }
      if (sum != Complex(0, 0)) {
        expected_result.InsertElement(i, sum, j);
      }
    }
  }

  // Сравнение результатов
  bool are_results_equal = yasakova_t_sparse_matrix_multiplication_seq::AreMatricesEqual(final_result, expected_result);
  EXPECT_TRUE(are_results_equal);
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_task_run) {
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_a(400, true, 400);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_b(400, true, 400);
  std::vector<Complex> input_data = {};
  std::vector<Complex> vector_a;
  std::vector<Complex> vector_b;
  std::vector<Complex> result_vector(sparse_matrix_a.columnCount * sparse_matrix_b.rowCount * 100, 0);

  for (unsigned int row = 0; row < 150; row++) {
    for (unsigned int col = 0; col < 150; col++) {
      sparse_matrix_a.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)),
                                    static_cast<int>(col));
    }
  }
  for (unsigned int row = 50; row < 140; row++) {
    for (unsigned int col = 50; col < 150; col++) {
      sparse_matrix_b.InsertElement(static_cast<int>(row), Complex(-50 + (rand() % 50), -50 + (rand() % 50)),
                                    static_cast<int>(col));
    }
  }
  vector_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_a);
  vector_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_b);
  input_data.reserve(vector_a.size() + vector_b.size());
  input_data.insert(input_data.end(), vector_a.begin(), vector_a.end());
  input_data.insert(input_data.end(), vector_b.begin(), vector_b.end());

  // Initialize task data structure
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_vector.data()));
  task_data->outputs_count.emplace_back(result_vector.size());

  // Create Task
  auto sequential_task =
      std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(task_data);

  // Create Performance attributes
  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performance_attributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  // Create Performance analyzer
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(sequential_task);
  performance_analyzer->TaskRun(performance_attributes, performance_results);
  ppc::core::Perf::PrintPerfStatistic(performance_results);
  yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS final_result =
      yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(result_vector);
}