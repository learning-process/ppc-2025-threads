#pragma once

#include <boost/mpi.hpp>
#include <cstddef>
#include <vector>
#include "core/task/include/task.hpp"

namespace kharin_m_multidimensional_integral_calc_all {

class TaskALL : public ppc::core::Task {
 public:
  explicit TaskALL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;        // Входные данные, только в rank 0
  std::vector<double> local_input_;  // Локальная часть входных данных для каждого процесса
  std::vector<size_t> grid_sizes_;   // Размеры сетки в каждом измерении
  std::vector<double> step_sizes_;   // Шаги интегрирования в каждом измерении
  double output_result_;             // Результат вычисления интеграла, только в rank 0
  boost::mpi::communicator world_;   // Коммуникатор MPI
  size_t num_threads_{1};            // Количество потоков в каждом процессе
};

}  // namespace kharin_m_multidimensional_integral_calc_all