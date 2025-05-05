#include "stl/anufriev_d_integrals_simpson/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}
}  // namespace

namespace anufriev_d_integrals_simpson_stl {

double IntegralsSimpsonSTL::FunctionN(const std::vector<double>& coords) const {
  switch (func_code_) {
    case 0: {
      double s = 0.0;
      for (double c : coords) {
        s += c * c;
      }
      return s;
    }
    case 1: {
      double val = 1.0;
      for (size_t i = 0; i < coords.size(); i++) {
        if (i % 2 == 0) {
          val *= std::sin(coords[i]);
        } else {
          val *= std::cos(coords[i]);
        }
      }
      return val;
    }
    default:
      return 0.0;
  }
}

double IntegralsSimpsonSTL::RecursiveSimpsonSum(int dim_index, std::vector<int>& idx,
                                                const std::vector<double>& steps) const {
  if (dim_index == dimension_) {
    double coeff = 1.0;
    std::vector<double> coords(dimension_);
    for (int d = 0; d < dimension_; ++d) {
      if (d < 0 || static_cast<size_t>(d) >= idx.size() || static_cast<size_t>(d) >= steps.size() ||
          static_cast<size_t>(d) >= a_.size()) {
        throw std::out_of_range("Index out of bounds in RecursiveSimpsonSum inner loop");
      }
      coords[d] = a_[d] + idx[d] * steps[d];
      if (d < 0 || static_cast<size_t>(d) >= n_.size()) {
        throw std::out_of_range("Index out of bounds for n_ in RecursiveSimpsonSum");
      }
      coeff *= SimpsonCoeff(idx[d], n_[d]);
    }
    return coeff * FunctionN(coords);
  }
  double sum = 0.0;
  if (dim_index < 0 || static_cast<size_t>(dim_index) >= n_.size() || static_cast<size_t>(dim_index) >= idx.size()) {
    throw std::out_of_range("Index out of bounds before RecursiveSimpsonSum loop");
  }
  for (int i = 0; i <= n_[dim_index]; ++i) {
    idx[dim_index] = i;
    sum += RecursiveSimpsonSum(dim_index + 1, idx, steps);
  }
  return sum;
}

bool IntegralsSimpsonSTL::PreProcessingImpl() {
  if (task_data->inputs.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t in_size_bytes = task_data->inputs_count[0];
  size_t num_doubles = in_size_bytes / sizeof(double);

  if (num_doubles < 1) {
    return false;
  }

  int d = static_cast<int>(in_ptr[0]);
  if (d < 1) {
    return false;
  }

  size_t needed_count = static_cast<size_t>(3 * d) + 2;
  if (num_doubles < needed_count) {
    return false;
  }

  dimension_ = d;
  a_.resize(dimension_);
  b_.resize(dimension_);
  n_.resize(dimension_);

  int idx_ptr = 1;
  for (int i = 0; i < dimension_; i++) {
    a_[i] = in_ptr[idx_ptr++];
    b_[i] = in_ptr[idx_ptr++];
    n_[i] = static_cast<int>(in_ptr[idx_ptr++]);
    if (n_[i] <= 0 || (n_[i] % 2) != 0) {
      return false;
    }
  }

  func_code_ = static_cast<int>(in_ptr[idx_ptr]);
  result_ = 0.0;
  return true;
}

bool IntegralsSimpsonSTL::ValidationImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < 1) {
    return false;
  }
  return true;
}

bool IntegralsSimpsonSTL::RunImpl() {
  if (dimension_ < 1) {
    return false;
  }

  std::vector<double> steps(dimension_);
  for (int i = 0; i < dimension_; i++) {
    if (n_[i] == 0) {
      return false;
    }
    steps[i] = (b_[i] - a_[i]) / n_[i];
  }

  std::vector<int> first_dim_indices(n_[0] + 1);
  std::iota(first_dim_indices.begin(), first_dim_indices.end(), 0);

  double total_sum = 0.0;
  try {
    total_sum = std::transform_reduce(
        std::execution::par, first_dim_indices.begin(), first_dim_indices.end(), 0.0, std::plus<double>(),
        [&](int i) -> double {
          if (dimension_ < 1) {
            throw std::logic_error("IntegralsSimpsonSTL::RunImpl: dimension_ < 1 inside parallel lambda!");
          }
          std::vector<int> local_idx(dimension_);
          local_idx[0] = i;
          return RecursiveSimpsonSum(1, local_idx, steps);
        });
  } catch (const std::exception& e) {
    (void)e;
    return false;
  }

  double coeff = 1.0;
  for (int i = 0; i < dimension_; i++) {
    coeff *= steps[i] / 3.0;
  }

  result_ = coeff * total_sum;
  return true;
}

bool IntegralsSimpsonSTL::PostProcessingImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  out_ptr[0] = result_;
  return true;
}

}  // namespace anufriev_d_integrals_simpson_stl