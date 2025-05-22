#include <gtest/gtest.h>
#include <omp.h>  // Добавлен заголовочный файл OpenMP

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/yasakova_t_sparse_matrix_multiplication/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(yasakova_t_sparse_matrix_mult_all, test_matmul_only_real) {
  // Инициализация MPI
  boost::mpi::communicator world;
  
  // Создание данных
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS a(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS b(3, 3);
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(3, 3);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  // Заполнение матриц
  a.AddValue(0, Complex(1, 0), 0);
  a.AddValue(0, Complex(2, 0), 2);
  a.AddValue(1, Complex(3, 0), 1);
  a.AddValue(2, Complex(4, 0), 0);
  a.AddValue(2, Complex(5, 0), 1);

  b.AddValue(0, Complex(6, 0), 1);
  b.AddValue(1, Complex(7, 0), 0);
  b.AddValue(2, Complex(8, 0), 2);

  // Подготовка входных данных
  in_a = yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(a);
  in_b = yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  in.insert(in.end(), in_a.begin(), in_a.end());
  in.insert(in.end(), in_b.begin(), in_b.end());

  // Ожидаемый результат
  c.AddValue(0, Complex(6, 0), 1);
  c.AddValue(0, Complex(16, 0), 2);
  c.AddValue(1, Complex(21, 0), 0);
  c.AddValue(2, Complex(24, 0), 1);
  c.AddValue(2, Complex(35, 0), 0);

  // Создание задачи
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Выполнение теста
  yasakova_t_sparse_matrix_mult_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  
  // Установка числа потоков OpenMP
  omp_set_num_threads(omp_get_max_threads());
  
  test_task_all.Run();
  test_task_all.PostProcessing();

  // Проверка результатов только на нулевом процессе
  if (world.rank() == 0) {
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS res =
        yasakova_t_sparse_matrix_mult_all::ParseVectorIntoMatrix(out);
    ASSERT_TRUE(yasakova_t_sparse_matrix_mult_all::CheckMatrixesEquality(res, c));
  }
}