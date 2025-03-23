#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/yasakova_t_sparse_matrix_multiplication/include/ops_seq.hpp"

// Эталонная реализация умножения матриц (простое поэлементное умножение)
std::vector<Complex> SimpleMatrixMultiplication(const yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS& a,
  const yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS& b) {
  std::vector<Complex> result(a.rowCount * b.columnCount, 0);
  for (size_t i = 0; i < a.rowCount; ++i) {
    for (size_t j = 0; j < b.columnCount; ++j) {
      for (size_t k = 0; k < a.columnCount; ++k) {
        result[i * b.columnCount + j] += a.GetElement(i, k) * b.GetElement(k, j);
      }
    }
  }
  return result;
}

// Проверка корректности результата
void CheckCorrectness(const std::vector<Complex>& result, const std::vector<Complex>& expected) {
  ASSERT_EQ(result.size(), expected.size()) << "Размеры результата и ожидаемого значения не совпадают!";
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_NEAR(result[i].real(), expected[i].real(), 1e-6) << "Ошибка в элементе " << i << " (real часть)";
    ASSERT_NEAR(result[i].imag(), expected[i].imag(), 1e-6) << "Ошибка в элементе " << i << " (imag часть)";
  }
}

TEST(yasakova_t_sparse_matrix_multiplication_seq, test_pipeline_run) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-50, 49);

    yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_a(400, true, 400);
    yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS sparse_matrix_b(400, true, 400);

    for (unsigned int row = 0; row < 150; row++) {
        for (unsigned int col = 0; col < 150; col++) {
            sparse_matrix_a.InsertElement(static_cast<int>(row), Complex(dis(gen), dis(gen)), static_cast<int>(col));
        }
    }
    for (unsigned int row = 50; row < 140; row++) {
        for (unsigned int col = 50; col < 150; col++) {
            sparse_matrix_b.InsertElement(static_cast<int>(row), Complex(dis(gen), dis(gen)), static_cast<int>(col));
        }
    }

    std::vector<Complex> vector_a = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_a);
    std::vector<Complex> vector_b = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(sparse_matrix_b);
    std::vector<Complex> input_data;
    input_data.reserve(vector_a.size() + vector_b.size());
    input_data.insert(input_data.end(), vector_a.begin(), vector_a.end());
    input_data.insert(input_data.end(), vector_b.begin(), vector_b.end());

    std::vector<Complex> result_vector(sparse_matrix_a.columnCount * sparse_matrix_b.rowCount, 0);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data->outputs_count.emplace_back(result_vector.size());

    auto sequential_task =
        std::make_shared<yasakova_t_sparse_matrix_multiplication_seq::SequentialMatrixMultiplicationTest>(task_data);

    auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
    performance_attributes->num_running = 10;
    const auto start_time = std::chrono::high_resolution_clock::now();
    performance_attributes->current_timer = [&] {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
        return static_cast<double>(duration) * 1e-9;
    };

    auto performance_results = std::make_shared<ppc::core::PerfResults>();
    auto performance_analyzer = std::make_shared<ppc::core::Perf>(sequential_task);
    performance_analyzer->PipelineRun(performance_attributes, performance_results);
    ppc::core::Perf::PrintPerfStatistic(performance_results);

    // Получаем результат от вашего алгоритма
    yasakova_t_sparse_matrix_multiplication_seq::SparseMatrixCRS final_result =
        yasakova_t_sparse_matrix_multiplication_seq::ConvertVectorToMatrix(result_vector);

    // Вычисляем ожидаемый результат с помощью эталонной реализации
    std::vector<Complex> expected_result = SimpleMatrixMultiplication(sparse_matrix_a, sparse_matrix_b);

    // Преобразуем результат вашего алгоритма в вектор для сравнения
    std::vector<Complex> result_vector_final = yasakova_t_sparse_matrix_multiplication_seq::ConvertMatrixToVector(final_result);

    // Проверяем корректность
    CheckCorrectness(result_vector_final, expected_result);
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