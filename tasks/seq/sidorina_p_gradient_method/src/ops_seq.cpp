#include "seq/sidorina_p_gradient_method/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool sidorina_p_gradient_method_seq::GradientMethod::PreProcessingImpl() {
  size = *reinterpret_cast<int*>(task_data->inputs[0]);
  tolerance = *reinterpret_cast<double*>(task_data->inputs[1]);
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  int a_size = task_data->inputs_count[2];
  a.assign(a_ptr, a_ptr + a_size);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
  int b_size = task_data->inputs_count[3];
  b.assign(b_ptr, b_ptr + b_size);
  auto* solution_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  int solution_size = task_data->inputs_count[4];
  solution.assign(solution_ptr, solution_ptr + solution_size);
  result.resize(size);
  return true;
}

bool sidorina_p_gradient_method_seq::GradientMethod::ValidationImpl() {
  if (*reinterpret_cast<int*>(task_data->inputs[0]) <= 0 || *reinterpret_cast<int*>(task_data->inputs[1]) < 0  ||
      static_cast<int>(task_data->inputs_count[2]) <= 0 ||
      static_cast<int>(task_data->inputs_count[3]) <= 0 ||
      static_cast<int>(task_data->inputs_count[4]) <= 0 )
    return false;

  if (*reinterpret_cast<int*>(task_data->inputs[0]) !=
      static_cast<int>(task_data->inputs_count[3]) || *reinterpret_cast<int*>(task_data->inputs[0]) != static_cast<int>(task_data->inputs_count[4]) ||
      *reinterpret_cast<int*>(task_data->inputs[0]) * *reinterpret_cast<int*>(task_data->inputs[0]) !=
          static_cast<int>(task_data->inputs_count[2]))
    return false;

  if (task_data->inputs_count.size() < 5 || task_data->inputs.size() < 5 || task_data->outputs.empty()) return false;

  const auto* matrix = reinterpret_cast<const double*>(task_data->inputs[2]);

  return matrixSimmPositive(matrix, *reinterpret_cast<int*>(task_data->inputs[0]));
}

bool sidorina_p_gradient_method_seq::GradientMethod::RunImpl() {
  result = ConjugateGradientMethod(a, b, solution, tolerance, size);
  return true;
}

bool sidorina_p_gradient_method_seq::GradientMethod::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}
