#include "seq/poroshin_v_multi_integral_with_trapez_method/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

void poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::count_multi_integral_trapez_method_seq() {
  int dimensions = static_cast<int>(limits.size());
  std::vector<double> h(dimensions);
  for (int i = 0; i < dimensions; ++i) {
    h[i] = (limits[i].second - limits[i].first) / n[i];
  }

  double integral = 0.0;
  std::vector<double> vars(dimensions);

  std::vector<long long> indices(dimensions, 0);
  int flag = 1;
  while (flag == 1) {
    for (int i = 0; i < dimensions; ++i) {
      vars[i] = limits[i].first + indices[i] * h[i];
    }

    double weight = 1.0;
    for (int i = 0; i < dimensions; ++i) {
      weight *= (indices[i] == 0 || indices[i] == n[i]) ? 0.5 : 1.0;
    }
    integral += func(vars) * weight;

    int dim_ = 0;
    while (dim_ < dimensions) {
      indices[dim_]++;
      if (indices[dim_] <= n[dim_]) {
        break;
      }
      indices[dim_] = 0;
      dim_++;
      if (dim_ == dimensions) {
        flag = 0;
        break;
      }
    }
  }

  double volume = 1.0;
  for (int i = 0; i < dimensions; ++i) {
    volume *= h[i];
  }

  res = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::PreProcessingImpl() {
  n.resize(dim);
  limits.resize(dim);
  for (size_t i = 0; i < dim; i++) {
    n[i] = reinterpret_cast<int *>(task_data->inputs[0])[i];
    limits[i].first = reinterpret_cast<double *>(task_data->inputs[1])[i];
    limits[i].second = reinterpret_cast<double *>(task_data->inputs[2])[i];
  }
  res = 0;
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim);
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::RunImpl() {
  count_multi_integral_trapez_method_seq();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res;
  return true;
}