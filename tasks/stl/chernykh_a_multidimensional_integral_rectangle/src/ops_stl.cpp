#include "stl/chernykh_a_multidimensional_integral_rectangle/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace chernykh_a_multidimensional_integral_rectangle_stl {

double Dimension::GetLowerBound() const { return lower_bound_; }

double Dimension::GetUpperBound() const { return upper_bound_; }

int Dimension::GetStepsCount() const { return steps_count_; }

double Dimension::GetStepSize() const { return (upper_bound_ - lower_bound_) / steps_count_; }

bool Dimension::IsValid() const { return lower_bound_ < upper_bound_ && steps_count_ > 0; }

bool STLTask::ValidationImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  return dims_size > 0 &&
         std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
}

bool STLTask::PreProcessingImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  dims_.assign(dims_ptr, dims_ptr + dims_size);
  return true;
}

bool STLTask::RunImpl() {
  int total_points = GetTotalPoints();
  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk_size = total_points / num_threads;

  auto threads = std::vector<std::thread>();
  auto thread_results = std::vector<double>(num_threads, 0.0);

  auto process_chunk = [&](int start, int end, int thread_index) -> void {
    auto thread_point = Point(dims_.size());
    for (int i = start; i < end; i++) {
      FillPoint(i, thread_point);
      thread_results[thread_index] += func_(thread_point);
    }
  };

  for (int i = 0; i < num_threads; i++) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? total_points : start + chunk_size;
    threads.emplace_back(process_chunk, start, end, i);
  }

  for (auto &thread : threads) {
    thread.join();
  }

  result_ = std::accumulate(thread_results.begin(), thread_results.end(), 0.0);
  result_ *= GetScalingFactor();
  return true;
}

bool STLTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

void STLTask::FillPoint(int index, Point &point) const {
  for (size_t i = 0; i < dims_.size(); i++) {
    int coordinate_index = index % dims_[i].GetStepsCount();
    point[i] = dims_[i].GetLowerBound() + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].GetStepsCount();
  }
}

int STLTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](int accum, const Dimension &dim) -> int { return accum * dim.GetStepsCount(); });
}

double STLTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_stl
