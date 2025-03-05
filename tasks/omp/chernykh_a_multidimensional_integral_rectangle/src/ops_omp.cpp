#include "omp/chernykh_a_multidimensional_integral_rectangle/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_omp {

bool Dimension::IsValid() const { return lower_bound < upper_bound && steps_count > 0; }

double Dimension::GetStepSize() const { return (upper_bound - lower_bound) / steps_count; }

bool OMPTask::ValidationImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  return dims_size > 0 &&
         std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
}

bool OMPTask::PreProcessingImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  dims_.assign(dims_ptr, dims_ptr + dims_size);
  return true;
}

bool OMPTask::RunImpl() {
  int total_points = GetTotalPoints();
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) default(none) shared(total_points)
  for (int i = 0; i < total_points; i++) {
    sum += func_(GetPoint(i));
  }
  result_ = sum * GetScalingFactor();
  return true;
}

bool OMPTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

int OMPTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](const int accum, const Dimension &dim) -> int { return accum * dim.steps_count; });
}

Point OMPTask::GetPoint(int index) const {
  auto point = Point(dims_.size());
  for (size_t i = 0; i < point.size(); i++) {
    int coordinate_index = index % dims_[i].steps_count;
    point[i] = dims_[i].lower_bound + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].steps_count;
  }
  return point;
}

double OMPTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](const double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_omp
