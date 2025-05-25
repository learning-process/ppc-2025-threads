#include <gtest/gtest.h>
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "all/yasakova_t_sparse_matrix_multiplication/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS GenMatrix(
    unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
    unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value) {
  if (left_border_row > right_border_row || left_border_col > right_border_col || right_border_row > num_rows ||
      right_border_col > num_cols || min_value > max_value) {
    throw("ERROR!");
  }
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a((int)num_rows, (int)num_cols);
  for (unsigned int i = left_border_row; i < right_border_row; i++) {
    for (unsigned int j = left_border_col; j < right_border_col; j++) {
      a.AddValue((int)i, Complex(min_value + (rand() % max_value), min_value + (rand() % max_value)), (int)j);
    }
  }
  return a;
}
}  // namespace

TEST(yasakova_t_sparse_matrix_mult_task_all, test_pipeline_run) {
  boost::mpi::communicator world;
  srand(time(nullptr));

  // Инициализация матриц
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(400, 400);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(400, 400);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  b = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);
  
  // Подготовка входных данных
  in_a = yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(a);
  in_b = yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  // Настройка задачи
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Создание задачи
  auto test_task_all = std::make_shared<yasakova_t_sparse_matrix_mult_all::TestTaskALL>(task_data_all);

  // Настройка производительности
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Настройка OpenMP
  omp_set_num_threads(omp_get_max_threads());

  // Запуск теста производительности
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  // Вывод результатов только на нулевом процессе
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res =
        yasakova_t_sparse_matrix_mult_all::ParseVectorIntoMatrix(out);
  }
}

TEST(yasakova_t_sparse_matrix_mult_task_all, test_task_run) {
  boost::mpi::communicator world;
  srand(time(nullptr));

  // Инициализация матриц
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(400, 400);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(400, 400);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a = ::GenMatrix(400, 400, 0, 150, 0, 150, -100, 100);
  b = ::GenMatrix(400, 400, 50, 140, 50, 150, -100, 100);

  // Подготовка входных данных
  in_a = yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(a);
  in_b = yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  // Настройка задачи
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Создание задачи
  auto test_task_all = std::make_shared<yasakova_t_sparse_matrix_mult_all::TestTaskALL>(task_data_all);

  // Настройка производительности
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Настройка OpenMP
  omp_set_num_threads(omp_get_max_threads());

  // Запуск теста производительности
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  // Вывод результатов только на нулевом процессе
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res =
        yasakova_t_sparse_matrix_mult_all::ParseVectorIntoMatrix(out);
  }
}