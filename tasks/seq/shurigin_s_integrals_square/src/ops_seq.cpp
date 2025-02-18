#include "seq/shurigin_s_integrals_square/include/ops_seq.hpp"

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_seq {

Integral::Integral(std::shared_ptr<ppc::core::TaskData> task_data)
    : Task(task_data),
      down_limit_(0.0),
      up_limit_(0.0),
      count_(0),
      result_(0.0),
      func_(nullptr),
      task_data_(std::move(task_data)) {}

bool Integral::PreProcessingImpl() {
  try {
    if (!task_data_ || task_data_->inputs.empty() || task_data_->inputs[0] == nullptr) {
      throw std::invalid_argument("Invalid input data.");
    }

    auto* inputs = reinterpret_cast<double*>(task_data_->inputs[0]);

    if (inputs == nullptr) {
      throw std::invalid_argument("Pointer to inputs is null.");
    }

    down_limit_ = inputs[0];
    up_limit_ = inputs[1];
    count_ = static_cast<int>(inputs[2]);

    if (count_ <= 0) {
      throw std::invalid_argument("Number of intervals must be positive.");
    }
    if (up_limit_ <= down_limit_) {
      throw std::invalid_argument("Upper limit must be greater than lower limit.");
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
    if (task_data_->inputs_count[0] != 3 * sizeof(double)) {
      throw std::invalid_argument("Expected 3 double values in input data.");
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

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("Function is not set. Use SetFunction() to set the function.");
    }
    result_ = Compute(func_, down_limit_, up_limit_, count_);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in RunImpl: " << e.what() << '\n';
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

void Integral::SetFunction(const std::function<double(double)>& func) { func_ = func; }

double Integral::Compute(const std::function<double(double)>& f, double a, double b, int n) {
  double step = (b - a) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = a + ((i + 0.5) * step);
    area += f(x) * step;
  }

  return area;
}

}  // namespace shurigin_s_integrals_square_seq
