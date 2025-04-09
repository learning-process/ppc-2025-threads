#include "tbb/anufriev_d_integrals_simpson/include/ops_tbb.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <numeric> // For std::accumulate
#include <stdexcept> // For potential errors
#include <iostream> // For potential debug output

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace {

// Simpson coefficient remains the same
int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}
} // namespace

namespace anufriev_d_integrals_simpson_tbb {

// FunctionN remains the same
double IntegralsSimpsonTBB::FunctionN(const std::vector<double>& coords) const {
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
      // Consider logging or throwing an error for unknown func_code_
      // std::cerr << "Warning: Unknown function code: " << func_code_ << std::endl;
      return 0.0;
  }
}

// PreProcessingImpl remains the same, just class name changes
bool IntegralsSimpsonTBB::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
      // std::cerr << "Error: Input data pointer is null or inputs vector is empty." << std::endl;
      return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t in_size_bytes = task_data->inputs_count[0];
  size_t num_doubles = in_size_bytes / sizeof(double);

  if (num_doubles < 1) {
    // std::cerr << "Error: Not enough data for dimension." << std::endl;
    return false;
  }

  // Check for potential overflow if dimension is excessively large
  // although unlikely with double representation.
  int d = static_cast<int>(in_ptr[0]);
  if (d <= 0) { // Dimension must be positive
    // std::cerr << "Error: Dimension must be positive, got " << d << std::endl;
    return false;
  }

  // Calculate needed count safely
  size_t required_elements = 1 + static_cast<size_t>(3 * d) + 1;
  if (num_doubles < required_elements) {
    // std::cerr << "Error: Not enough input data for dimension " << d
              // << ". Need " << required_elements << " doubles, got " << num_doubles << "." << std::endl;
    return false;
  }

  dimension_ = d;
  try {
      a_.resize(dimension_);
      b_.resize(dimension_);
      n_.resize(dimension_);
  } catch (const std::bad_alloc& e) {
      std::cerr << "Error: Failed to allocate memory for dimension " << dimension_ << ": " << e.what() << std::endl;
    return false;
  }


  int idx_ptr = 1;
  for (int i = 0; i < dimension_; i++) {
    a_[i] = in_ptr[idx_ptr++];
    b_[i] = in_ptr[idx_ptr++];
    // Check for potential non-integer values before casting
    double n_double = in_ptr[idx_ptr++];
    if (std::floor(n_double) != n_double || n_double > static_cast<double>(std::numeric_limits<int>::max())) {
        // std::cerr << "Error: Number of steps n[" << i << "] must be an integer." << std::endl;
        return false;
    }
    n_[i] = static_cast<int>(n_double);

    if (n_[i] <= 0 || (n_[i] % 2) != 0) {
        // std::cerr << "Error: n[" << i << "] = " << n_[i] << " must be a positive even integer." << std::endl;
      return false;
    }
    if (a_[i] > b_[i]) {
        // std::cerr << "Warning: Lower bound a[" << i << "] = " << a_[i]
        //           << " is greater than upper bound b[" << i << "] = " << b_[i] << "." << std::endl;
        // Depending on requirements, this could be an error or just allowed (integral might be negative)
    }
  }

  // Check if func_code can be safely cast to int
  double func_code_double = in_ptr[idx_ptr];
  if (std::floor(func_code_double) != func_code_double || func_code_double > static_cast<double>(std::numeric_limits<int>::max()) || func_code_double < static_cast<double>(std::numeric_limits<int>::min())) {
     // std::cerr << "Error: Function code must be an integer." << std::endl;
     return false;
  }
  func_code_ = static_cast<int>(func_code_double);

  result_ = 0.0;

  return true;
}

// ValidationImpl remains the same, just class name changes
bool IntegralsSimpsonTBB::ValidationImpl() {
  // Check if output pointer and count are valid
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    // std::cerr << "Error: Output data pointer is null or outputs vector is empty." << std::endl;
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < sizeof(double)) {
    // std::cerr << "Error: Output buffer size is insufficient." << std::endl;
    return false;
  }
  // Optional: Could add validation for inputs here as well,
  // though PreProcessingImpl already does a thorough job.
  // For instance, check if dimension_ > 0 after potential PreProcessing.
  // return dimension_ > 0; // Example: ensure dimension was set correctly
  return true; // Basic validation passed
}


bool IntegralsSimpsonTBB::RunImpl() {
    std::vector<double> steps(dimension_);
    size_t total_points = 1; // Use size_t for potentially large number of points
    double coeff_mult = 1.0;

    for (int i = 0; i < dimension_; i++) {
        if (n_[i] == 0) { // Should be caught by PreProcessing, but double-check
            // std::cerr << "Error: n_[" << i << "] is zero." << std::endl;
            return false;
        }
        steps[i] = (b_[i] - a_[i]) / n_[i];
        coeff_mult *= steps[i] / 3.0;
        // Check for potential overflow when calculating total_points
        size_t points_in_dim = static_cast<size_t>(n_[i]) + 1;
        if (total_points > std::numeric_limits<size_t>::max() / points_in_dim) {
            // std::cerr << "Error: Total number of points exceeds size_t capacity." << std::endl;
            return false; // Overflow would occur
        }
        total_points *= points_in_dim;
    }

     // Use thread-local storage for coords to avoid repeated allocations if performance-critical
     // For simplicity here, we create it inside the lambda.
    // std::vector<double> local_coords(dimension_);

    double total_sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, total_points), // Range of linear indices
        0.0, // Identity value for summation
        [&](const tbb::blocked_range<size_t>& r, double running_sum) {
            // Thread-local vector for coordinates to avoid race conditions
            std::vector<double> coords(dimension_);
            // Thread-local vector for multidimensional indices
            std::vector<int> current_idx(dimension_);

            for (size_t k = r.begin(); k != r.end(); ++k) {
                double current_coeff_prod = 1.0;
                size_t current_k = k; // Temporary variable for index calculation

                // Convert linear index 'k' to multidimensional index 'current_idx'
                // and calculate coordinates and Simpson coefficients product
                //size_t product_of_dimensions = 1; // To help with index calculation
                for (int dim = 0; dim < dimension_; ++dim) {
                     size_t points_in_this_dim = static_cast<size_t>(n_[dim]) + 1;
                     size_t index_in_this_dim = current_k % points_in_this_dim;
                     current_idx[dim] = static_cast<int>(index_in_this_dim);
                     current_k /= points_in_this_dim;

                     coords[dim] = a_[dim] + current_idx[dim] * steps[dim];
                     current_coeff_prod *= SimpsonCoeff(current_idx[dim], n_[dim]);
                }

                running_sum += current_coeff_prod * FunctionN(coords);
            }
            return running_sum;
        },
        [](double x, double y) { // Reduction operation: sum
            return x + y;
        }
    );

    result_ = coeff_mult * total_sum;
    return true;
}


// PostProcessingImpl remains the same, just class name changes
bool IntegralsSimpsonTBB::PostProcessingImpl() {
  try {
      auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
      out_ptr[0] = result_;
  } catch (const std::exception& e) {
      std::cerr << "Error during PostProcessing: " << e.what() << std::endl;
    return false; // Indicate failure if access causes exception
  }
  return true;
}

} // namespace anufriev_d_integrals_simpson_tbb