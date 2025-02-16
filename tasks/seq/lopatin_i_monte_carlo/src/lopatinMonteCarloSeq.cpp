#include "seq/lopatin_i_monte_carlo/include/lopatinMonteCarloSeq.hpp"

namespace lopatin_i_monte_carlo_seq {

bool TestTaskSequential::ValidationImpl() {
  const bool outputs_valid = !task_data->outputs_count.empty() && task_data->outputs_count[0] == 1;
  const bool inputs_valid = task_data->inputs_count.size() == 2 &&
                            (task_data->inputs_count[0] % 2 == 0) &&  // odd num of bounds
                            task_data->inputs_count[1] == 1;          // iterations num
  return outputs_valid && inputs_valid;
}

bool TestTaskSequential::PreProcessingImpl() {
  auto* bounds_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t bounds_size = task_data->inputs_count[0];
  integrationBounds.resize(bounds_size);
  std::copy(bounds_ptr, bounds_ptr + bounds_size, integrationBounds.begin());

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  iterations = *iter_ptr;
  return true;
}

bool TestTaskSequential::RunImpl() {
  const int D = integrationBounds.size() / 2; // dimensions
  std::mt19937 rnd(12);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  result = 0.0;
  for (int i = 0; i < iterations; ++i) {
    std::vector<double> point(D);
    for (int j = 0; j < D; ++j) {
      const double min = integrationBounds[2 * j];
      const double max = integrationBounds[2 * j + 1];
      point[j] = min + (max - min) * dis(rnd);
    }
    result += integrand(point);
  }

  // volume of integration region
  double volume = 1.0;
  for (int j = 0; j < D; ++j) {
    volume *= (integrationBounds[2 * j + 1] - integrationBounds[2 * j]);
  }
  result = (result / iterations) * volume;

  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result;
  return true;
}

}  // namespace lopatin_i_monte_carlo_seq
