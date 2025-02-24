#include "seq/chernykh_a_multidimensional_integral_rectangle/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_seq {

bool Dimension::IsValid() const { return lower_bound_ < upper_bound_ && steps_count_ > 0; }

double Dimension::GetStepSize() const { return (upper_bound_ - lower_bound_) / steps_count_; }

bool SequentialTask::ValidationImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  return dims_size > 0 &&
         std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
}

bool SequentialTask::PreProcessingImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  dims_.assign(dims_ptr, dims_ptr + dims_size);
  return true;
}

bool SequentialTask::RunImpl() {
  int total_points = GetTotalPoints();
  for (int i = 0; i < total_points; i++) {
    result_ += func_(GetPoint(i));
  }
  result_ *= GetScalingFactor();
  return true;
}

bool SequentialTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

int SequentialTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](const int accum, const Dimension &dim) -> int { return accum * dim.steps_count_; });
}

Point SequentialTask::GetPoint(int index) const {
  auto point = Point(dims_.size());
  for (size_t i = 0; i < point.size(); i++) {
    int coordinate_index = index % dims_[i].steps_count_;
    point[i] = dims_[i].lower_bound_ + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].steps_count_;
  }
  return point;
}

double SequentialTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](const double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
