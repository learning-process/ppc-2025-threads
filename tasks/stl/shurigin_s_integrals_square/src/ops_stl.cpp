#include "stl/shurigin_s_integrals_square/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace shurigin_s_integrals_square_stl {

Integral::Integral(std::shared_ptr<ppc::core::TaskData> task_data)
    : Task(task_data),
      down_limits_(1, 0.0),
      up_limits_(1, 0.0),
      counts_(1, 0),
      result_(0.0),
      func_(nullptr),
      dimensions_(1),
      task_data_(std::move(task_data)) {}

void Integral::SetFunction(const std::function<double(double)>& func) {
  func_ = [func](const std::vector<double>& point) { return func(point[0]); };
  dimensions_ = 1;
  down_limits_.resize(1);
  up_limits_.resize(1);
  counts_.resize(1);
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions) {
  func_ = func;
  dimensions_ = dimensions;
  down_limits_.resize(dimensions_);
  up_limits_.resize(dimensions_);
  counts_.resize(dimensions_);
}

bool Integral::PreProcessingImpl() {
  try {
    if (!task_data_ || task_data_->inputs.empty() || task_data_->inputs[0] == nullptr) {
      throw std::invalid_argument("Invalid input data.");
    }
    auto* inputs = reinterpret_cast<double*>(task_data_->inputs[0]);

    if (dimensions_ <= 0) {
      throw std::invalid_argument("Dimensions must be positive.");
    }

    down_limits_.resize(dimensions_);
    up_limits_.resize(dimensions_);
    counts_.resize(dimensions_);

    size_t num_input_doubles = task_data_->inputs_count[0] / sizeof(double);

    for (int i = 0; i < dimensions_; ++i) {
      size_t count_index = static_cast<size_t>(i) + (2 * static_cast<size_t>(dimensions_));
      size_t upper_limit_index = static_cast<size_t>(i) + static_cast<size_t>(dimensions_);
      size_t lower_limit_index = static_cast<size_t>(i);

      if (count_index >= num_input_doubles || upper_limit_index >= num_input_doubles ||
          lower_limit_index >= num_input_doubles) {
        throw std::out_of_range("Input data size is too small for the specified dimensions and required elements.");
      }
      down_limits_[i] = inputs[lower_limit_index];
      up_limits_[i] = inputs[upper_limit_index];
      counts_[i] = static_cast<int>(inputs[count_index]);

      if (counts_[i] <= 0) {
        throw std::invalid_argument("Number of intervals must be positive for all dimensions.");
      }
      if (up_limits_[i] <= down_limits_[i]) {
        throw std::invalid_argument("Upper limit must be greater than lower limit for all dimensions.");
      }
    }

    result_ = 0.0;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PreProcessingImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    if (!task_data_) {
      throw std::invalid_argument("task_data is null.");
    }
    if (task_data_->inputs_count.empty() || task_data_->outputs_count.empty()) {
      throw std::invalid_argument("Input or output counts are empty.");
    }

    if (dimensions_ <= 0) {
      throw std::invalid_argument("Dimensions must be positive for validation.");
    }

    size_t expected_input_size_bytes = 3 * dimensions_ * sizeof(double);
    if (task_data_->inputs_count[0] != expected_input_size_bytes) {
      throw std::invalid_argument("Input size mismatch: Expected " + std::to_string(expected_input_size_bytes) +
                                  " bytes (" + std::to_string(3 * dimensions_) + " doubles), but got " +
                                  std::to_string(task_data_->inputs_count[0]) + " bytes.");
    }

    if (task_data_->outputs_count[0] != sizeof(double)) {
      throw std::invalid_argument("Expected one double value in output data.");
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in ValidationImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::PostProcessingImpl() {
  try {
    if (!task_data_ || task_data_->outputs.empty() || task_data_->outputs[0] == nullptr) {
      throw std::invalid_argument("Invalid output data.");
    }
    auto* outputs = reinterpret_cast<double*>(task_data_->outputs[0]);
    outputs[0] = result_;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PostProcessingImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("Function is not set.");
    }

    if (dimensions_ <= 0) {
      throw std::invalid_argument("Dimensions must be positive to run.");
    }

    if (dimensions_ == 1) {
      return ComputeOneDimensionalStl();
    } else {
      std::vector<double> initial_point(dimensions_);
      result_ = ComputeParallelOuterLoop(func_, down_limits_, up_limits_, counts_, dimensions_, initial_point);
      return true;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error in RunImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::ComputeOneDimensionalStl() {
  const double lower = down_limits_[0];
  const double upper = up_limits_[0];
  const int num_intervals = counts_[0];

  if (num_intervals <= 0) {
    result_ = 0.0;
    return true;
  }

  const double step = (upper - lower) / num_intervals;
  const double half_step = 0.5 * step;
  const double base = lower + half_step;

  unsigned int num_workers = std::thread::hardware_concurrency();
  if (num_workers == 0) {
    num_workers = 1;
  }
  num_workers = std::min(num_workers, static_cast<unsigned int>(num_intervals));

  std::vector<std::thread> threads;
  threads.reserve(num_workers);
  std::vector<std::future<double>> futures;
  futures.reserve(num_workers);

  int intervals_per_worker = num_intervals / num_workers;
  int extra_intervals = num_intervals % num_workers;
  int current_start_index = 0;

  auto worker_task = [this, base, step](int start_idx, int end_idx, std::promise<double> promise) {
    double local_sum = 0.0;
    std::vector<double> point(1);
    for (int i = start_idx; i < end_idx; ++i) {
      point[0] = base + static_cast<double>(i) * step;
      local_sum += func_(point);
    }
    promise.set_value(local_sum);
  };

  for (unsigned int i = 0; i < num_workers; ++i) {
    int count_for_this_worker = intervals_per_worker + (i < static_cast<unsigned int>(extra_intervals) ? 1 : 0);
    if (count_for_this_worker == 0) continue;
    int current_end_index = current_start_index + count_for_this_worker;

    std::promise<double> promise;
    futures.emplace_back(promise.get_future());
    threads.emplace_back(worker_task, current_start_index, current_end_index, std::move(promise));

    current_start_index = current_end_index;
  }

  double total_sum = 0.0;
  for (auto& fut : futures) {
    total_sum += fut.get();
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  result_ = total_sum * step;
  return true;
}

double Integral::ComputeParallelOuterLoop(const std::function<double(const std::vector<double>&)>& f,
                                          const std::vector<double>& a, const std::vector<double>& b,
                                          const std::vector<int>& n, int dim, std::vector<double> initial_point) {
  if (dim <= 1) {
    throw std::logic_error("ComputeParallelOuterLoop called with invalid dimension <= 1");
  }

  const double step0 = (b[0] - a[0]) / n[0];
  const double base0 = a[0] + (0.5 * step0);
  const int num_intervals_outer = n[0];

  if (num_intervals_outer <= 0) {
    return 0.0;
  }

  unsigned int num_workers = std::thread::hardware_concurrency();
  if (num_workers == 0) {
    num_workers = 1;
  }
  num_workers = std::min(num_workers, static_cast<unsigned int>(num_intervals_outer));

  std::vector<std::thread> threads;
  threads.reserve(num_workers);
  std::vector<std::future<double>> futures;
  futures.reserve(num_workers);

  int intervals_per_worker = num_intervals_outer / num_workers;
  int extra_intervals = num_intervals_outer % num_workers;
  int current_start_index = 0;

  auto worker_task = [this, f, &a, &b, &n, dim, initial_point, base0, step0](int start_idx, int end_idx,
                                                                             std::promise<double> promise) {
    double local_sum = 0.0;
    std::vector<double> current_point = initial_point;
    for (int i = start_idx; i < end_idx; ++i) {
      current_point[0] = base0 + static_cast<double>(i) * step0;
      local_sum += ComputeSequentialRecursive(f, a, b, n, dim, current_point, 1);
    }
    promise.set_value(local_sum);
  };

  for (unsigned int i = 0; i < num_workers; ++i) {
    int count_for_this_worker = intervals_per_worker + (i < static_cast<unsigned int>(extra_intervals) ? 1 : 0);
    if (count_for_this_worker == 0) continue;
    int current_end_index = current_start_index + count_for_this_worker;

    std::promise<double> promise;
    futures.emplace_back(promise.get_future());
    threads.emplace_back(worker_task, current_start_index, current_end_index, std::move(promise));

    current_start_index = current_end_index;
  }

  double total_sum_outer = 0.0;
  for (auto& fut : futures) {
    total_sum_outer += fut.get();
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  return total_sum_outer * step0;
}

double Integral::ComputeSequentialRecursive(const std::function<double(const std::vector<double>&)>& f,
                                            const std::vector<double>& a, const std::vector<double>& b,
                                            const std::vector<int>& n, int dim, std::vector<double>& point,
                                            int current_dim) {
  if (current_dim == dim) {
    return f(point);
  }

  const double step = (b[current_dim] - a[current_dim]) / n[current_dim];
  const double base = a[current_dim] + (0.5 * step);
  double sum_over_dimension = 0.0;

  for (int i = 0; i < n[current_dim]; ++i) {
    point[current_dim] = base + static_cast<double>(i) * step;
    sum_over_dimension += ComputeSequentialRecursive(f, a, b, n, dim, point, current_dim + 1);
  }

  return sum_over_dimension * step;
}

}  // namespace shurigin_s_integrals_square_stl