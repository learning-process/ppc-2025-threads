#include "../include/integrate_seq.hpp"

#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "seq/khasanyanov_k_trapezoid_method/include/integrator.hpp"

using namespace khasanyanov_k_trapezoid_method_seq;

std::shared_ptr<ppc::core::TaskData> TrapezoidalMethodSequential::CreateTaskData(
    const IntegrateFunction &f, const IntegrateBounds &bounds,
    double precision,  // NOLINT(bugprone-easily-swappable-parameters)
    double *out) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = precision};
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&context));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
  task_data->outputs_count.emplace_back(1);
  return task_data;
}

bool TrapezoidalMethodSequential::ValidationImpl() {
  auto *data = reinterpret_cast<TaskContext *>(task_data->inputs[0]);
  return data != nullptr && !data->bounds.empty() && !task_data->outputs.empty();
}

bool TrapezoidalMethodSequential::PreProcessingImpl() {
  data_ = *reinterpret_cast<TaskContext *>(task_data->inputs[0]);
  return true;
}
bool TrapezoidalMethodSequential::RunImpl() {
  res_ = Integrator<kSequential>{}(data_.function, data_.bounds, data_.precision);
  return true;
}
bool TrapezoidalMethodSequential::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = res_;
  return true;
}