#pragma once
#include <mpi.h>  // MPI include

#include <cmath>
#include <functional>
#include <memory>
#include <string>  // For std::string in exceptions
#include <vector>

#include "core/task/include/task.hpp"  // Стандартный путь, если ваш проект так настроен

namespace shurigin_s_integrals_square_mpi {
class Integral : public ppc::core::Task {
 public:
  explicit Integral(std::shared_ptr<ppc::core::TaskData> task_data_param);  // Изменено имя параметра

  void SetFunction(const std::function<double(double)>& func);
  void SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions = 1);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> down_limits_;
  std::vector<double> up_limits_;
  std::vector<int> counts_;
  double result_;
  std::function<double(const std::vector<double>&)> func_;
  int dimensions_;

  // Возвращаем локальный член task_data_
  std::shared_ptr<ppc::core::TaskData> task_data_;  // <--- ВОТ ЭТО ВАЖНО

  // MPI specific members
  int mpi_rank_;
  int mpi_world_size_;

  // Helper methods for computation
  double ComputeOneDimensionalOMP(const std::function<double(const std::vector<double>&)>& f, double a_local,
                                  double b_local, int n_local);

  double ComputeOuterParallelInnerSequential(const std::function<double(const std::vector<double>&)>& f,
                                             double a0_local, double b0_local, int n0_local,
                                             const std::vector<double>& full_a, const std::vector<double>& full_b,
                                             const std::vector<int>& full_n, int total_dims);

  double ComputeSequentialRecursive(const std::function<double(const std::vector<double>&)>& f,
                                    const std::vector<double>& a, const std::vector<double>& b,
                                    const std::vector<int>& n, int total_dims, std::vector<double>& point,
                                    int current_dim_idx);
};
}  // namespace shurigin_s_integrals_square_mpi