#include "seq/kholin_k_multidimensional_integrals_rectangle/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

double kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::Integrate(
    const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double>& f_values, int curr_index_dim, double dim, double n) {
  if (curr_index_dim == static_cast<int>(dim)) {
    return f(f_values);
  }

  double sum = 0.0;
  for (double i = 0; i < n; ++i) {
    f_values[curr_index_dim] = l_limits[curr_index_dim] + (i + 0.5) * h[curr_index_dim];
    sum += Integrate(f, l_limits, u_limits, h, f_values, curr_index_dim + 1, dim, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::IntegrateWithRectangleMethod(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, double dim, double n) {
  std::vector<double> h(dim);
  for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }

  return Integrate(f, l_limits, u_limits, h, f_values, 0, dim, n);
}

double kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::RunMultistepSchemeMethodRectangle(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, double dim, double n, double epsilon) {
  double i_n = IntegrateWithRectangleMethod(f, f_values, l_limits, u_limits, dim, n);
  double i_2n = 0.0;
  double delta = 0;
  do {
    n *= 2;
    i_2n = IntegrateWithRectangleMethod(f, f_values, l_limits, u_limits, dim, n);
    delta = std::fabs(i_2n - i_n);
    i_n = i_2n;

  } while ((1.0 / 3) * delta >= epsilon);

  return i_2n;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  sz_values_ = task_data->inputs_count[0];
  sz_lower_limits_ = task_data->inputs_count[1];
  sz_upper_limits_ = task_data->inputs_count[2];

  auto* ptr_dim = reinterpret_cast<double*>(task_data->inputs[0]);
  dim_ = *ptr_dim;

  auto* ptr_f_values = reinterpret_cast<double*>(task_data->inputs[1]);
  f_values_.assign(ptr_f_values, ptr_f_values + sz_values_);

  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[2]);
  f_ = *ptr_f;

  auto* ptr_lower_limits = reinterpret_cast<double*>(task_data->inputs[3]);
  lower_limits_.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits_);

  auto* ptr_upper_limits = reinterpret_cast<double*>(task_data->inputs[4]);
  upper_limits_.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits_);

  auto* ptr_epsilon = reinterpret_cast<double*>(task_data->inputs[5]);
  epsilon_ = *ptr_epsilon;

  auto* ptr_start_n = reinterpret_cast<double*>(task_data->inputs[6]);
  start_n_ = *ptr_start_n;

  result_ = 0.0;
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[1] > 0U && task_data->inputs_count[2] > 0U;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::RunImpl() {
  result_ = RunMultistepSchemeMethodRectangle(f_, f_values_, lower_limits_, upper_limits_, dim_, start_n_, epsilon_);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}