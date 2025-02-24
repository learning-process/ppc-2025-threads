#include "seq/chernykh_a_multidimensional_integral_rectangle/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_seq {

bool SequentialTask::ValidationImpl() {
  auto *bounds_ptr = reinterpret_cast<Bounds *>(task_data->inputs[0]);
  uint32_t bounds_size = task_data->inputs_count[0];
  auto *steps_ptr = reinterpret_cast<Steps *>(task_data->inputs[1]);
  uint32_t steps_size = task_data->inputs_count[1];

  bool is_correct_bounds =
      std::all_of(bounds_ptr, bounds_ptr + bounds_size, [](const Bounds &b) -> bool { return b.first < b.second; });
  bool is_correct_steps = std::all_of(steps_ptr, steps_ptr + steps_size, [](const Steps &s) -> bool { return s > 0; });

  return bounds_size > 0 && is_correct_bounds && steps_size > 0 && is_correct_steps && bounds_size == steps_size;
}

bool SequentialTask::PreProcessingImpl() {
  auto *bounds_ptr = reinterpret_cast<Bounds *>(task_data->inputs[0]);
  uint32_t bounds_size = task_data->inputs_count[0];
  auto *steps_ptr = reinterpret_cast<Steps *>(task_data->inputs[1]);
  uint32_t steps_size = task_data->inputs_count[1];

  bounds_per_dim_.assign(bounds_ptr, bounds_ptr + bounds_size);
  steps_per_dim_.assign(steps_ptr, steps_ptr + steps_size);
  return true;
}

bool SequentialTask::RunImpl() {
  std::vector<double> step_size_per_dim = GetStepSizePerDim();
  int total_points = GetTotalPoints();

  double sum = 0.0;
  for (int p = 0; p < total_points; p++) {
    sum += func_(GetPoint(p, step_size_per_dim));
  }

  result_ = sum * GetScalingFactor(step_size_per_dim);
  return true;
}

bool SequentialTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

std::vector<double> SequentialTask::GetStepSizePerDim() const {
  auto step_size_per_dim = std::vector<double>(bounds_per_dim_.size());
  for (size_t i = 0; i < step_size_per_dim.size(); i++) {
    step_size_per_dim[i] = (bounds_per_dim_[i].second - bounds_per_dim_[i].first) / steps_per_dim_[i];
  }
  return step_size_per_dim;
}

int SequentialTask::GetTotalPoints() const {
  return std::accumulate(steps_per_dim_.begin(), steps_per_dim_.end(), 1, std::multiplies());
}

Point SequentialTask::GetPoint(int point_idx, const std::vector<double> &step_size_per_dim) const {
  auto point = std::vector<double>(bounds_per_dim_.size());
  for (size_t i = 0; i < point.size(); i++) {
    int coordinate_idx = point_idx % steps_per_dim_[i];
    point[i] = bounds_per_dim_[i].first + (coordinate_idx + 1) * step_size_per_dim[i];
    point_idx /= steps_per_dim_[i];
  }
  return point;
}

double SequentialTask::GetScalingFactor(const std::vector<double> &step_size_per_dim) {
  return std::accumulate(step_size_per_dim.begin(), step_size_per_dim.end(), 1.0, std::multiplies());
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
