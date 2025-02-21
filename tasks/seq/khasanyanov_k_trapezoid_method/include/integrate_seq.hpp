#ifndef _INTEGRATE_SEQ_HPP_
#define _INTEGRATE_SEQ_HPP_

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/khasanyanov_k_trapezoid_method/include/integrator.hpp"

namespace khasanyanov_k_trapezoid_method_seq {

struct TaskContext {
  IntegrateFunction function;
  IntegrateBounds bounds;
  double precision;
};

class TrapezoidalMethodSequential : public ppc::core::Task {
 public:
  explicit TrapezoidalMethodSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard("New task data igrored")]] static std::shared_ptr<ppc::core::TaskData> CreateTaskData(
      const IntegrateFunction&, const IntegrateBounds&, double, double*);

 private:
  TaskContext data_;
  double res_{};
};

}  // namespace khasanyanov_k_trapezoid_method_seq

#endif