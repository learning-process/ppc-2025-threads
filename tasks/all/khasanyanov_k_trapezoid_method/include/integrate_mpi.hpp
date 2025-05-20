#ifndef INTEGRATE_MPI_HPP
#define INTEGRATE_MPI_HPP

#include <functional>
#include <memory>
#include <utility>

#include "../include/integrator.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_trapezoid_method_mpi {

using IntegrationFunction = std::function<double(const std::vector<double> &)>;
using Bounds = std::pair<double, double>;
using IntegrationBounds = std::vector<Bounds>;

struct TaskContext {
  IntegrationFunction function;
  IntegrationBounds bounds;
  double precision;
};

class TrapezoidalMethodMPI : public ppc::core::Task {
 public:
  explicit TrapezoidalMethodMPI(ppc::core::TaskDataPtr task_data, IntegrationFunction function)
      : Task(std::move(task_data)), function_(std::move(function)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] double TrapezoidalMethodOmp(const IntegrationBounds &bounds, int steps);

  [[nodiscard]] static double CalculateCellVolume(const IntegrationBounds &bounds, int steps);

  static void CreateTaskData(std::shared_ptr<ppc::core::TaskData> &, TaskContext &context, double *);

 private:
  static const int kDefaultSteps, kMaxSteps;
  boost::mpi::communicator comm_;
  TaskContext data_;
  IntegrationFunction function_;
  double res_{};
};

}  // namespace khasanyanov_k_trapezoid_method_mpi

#endif