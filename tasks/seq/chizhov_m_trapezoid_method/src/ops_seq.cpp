#include <thread>

#include "seq/chizhov_m_trapezoid_method/include/ops_seq.hpp"

using namespace std::chrono_literals;

double chizhov_m_trapezoid_method_seq::Trapezoid_method(const Function f, int div,
                                                        int dim, std::vector<double>& lower_limits,
                                                        std::vector<double>& upper_limits) {
  std::vector<double> h(dim);
  std::vector<int> steps(dim);

  for (int i = 0; i < dim; i++) {
    steps[i] = div;
    h[i] = (upper_limits[i] - lower_limits[i]) / steps[i];
  }

  int total_nodes = 1;
  for (const auto& step : steps) {
    total_nodes *= (step + 1);
  }

  double result = 0.0;
  std::vector<double> point(dim);

  for (int i = 0; i < total_nodes; i++) {
    int temp = i;

    for (int j = 0; j < dim; j++) {
      int node_index = temp % (steps[j] + 1);
      point[j] = lower_limits[j] + node_index * h[j];
      temp /= (steps[j] + 1);
    }

    double weight = 1.0;
    for (int j = 0; j < dim; j++) {
      if (point[j] == lower_limits[j] || point[j] == upper_limits[j]) {
        weight *= 1.0;
      } else {
        weight *= 2.0;
      }
    }

    result += weight * f(point);
  }

  for (int i = 0; i < dim; i++) {
    result *= h[i] / 2.0;
  }

  return std::round(result * 100.0) / 100.0;

}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::PreProcessingImpl() {
  int* divisionsPtr = reinterpret_cast<int*>(task_data->inputs[0]);
  div = static_cast<size_t>(*divisionsPtr);

  int* dimensionPtr = reinterpret_cast<int*>(task_data->inputs[1]);
  dim = static_cast<size_t>(*dimensionPtr);

  auto* limitPtr = reinterpret_cast<double*>(task_data->inputs[2]);
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i+=2) {
    lower_limits.push_back(limitPtr[i]);
    upper_limits.push_back(limitPtr[i+1]);
  }
  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[3]);
  f = *ptr_f;

  return true;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::ValidationImpl() {
  int* divisionsPtr = reinterpret_cast<int*>(task_data->inputs[0]);
  int* dimensionPtr = reinterpret_cast<int*>(task_data->inputs[1]);
  if (*divisionsPtr <= 0 || *dimensionPtr <= 0) return false;
  if (task_data->inputs_count[2] % 2 != 0) return false;
  auto* limitPtr = reinterpret_cast<double*>(task_data->inputs[2]);
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
    if (limitPtr[i] >= limitPtr[i + 1]) return false;
  }

  return true;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::RunImpl() {
  res = Trapezoid_method(f, div, dim, lower_limits, upper_limits);

  return true;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res;

  return true;
}