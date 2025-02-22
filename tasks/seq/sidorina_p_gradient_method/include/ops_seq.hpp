#pragma once

#include <utility>
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace sidorina_p_gradient_method_seq {

inline std::vector<double> MultiplyMatrixByVector(const std::vector<double>& a, const std::vector<double>& vec,
                                                  int size) {
  std::vector<double> result(size, 0);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i] += a[(i * size) + j] * vec[j];
    }
  }
  return result;
}

inline double VectorNorm(const std::vector<double>& vec) {
  double sum = 0;
  for (double value : vec) {
    sum += std::pow(value, 2);
  }
  return std::sqrt(sum);
}

inline double Dot(const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double sum = 0;
  for (unsigned long i = 0; i < vec1.size(); i++) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
}

inline double Dot(const std::vector<double>& vec) {
  double sum = 0;
  for (unsigned long i = 0; i < vec.size(); i++) {
    sum += std::pow(vec[i], 2);
  }
  return sum;
}

inline std::vector<double> ConjugateGradientMethod(std::vector<double>& a, std::vector<double>& b,
                                                   std::vector<double> solution, double tolerance, int size) {
  std::vector<double> matrix_times_solution = MultiplyMatrixByVector(a, solution, size);

  auto residual = std::vector<double>(size);
  auto direction = std::vector<double>(size);

  std::transform(b.begin(), b.end(), residual.begin(),
                 [&matrix_times_solution](const double& val) { return val - matrix_times_solution[0]; });

  double residual_norm_squared = Dot(residual);
  if (std::sqrt(residual_norm_squared) < tolerance) {
    return solution;
  }
  direction = residual;
  std::vector<double> matrix_times_direction(size);
  while (std::sqrt(residual_norm_squared) > tolerance) {
    matrix_times_direction = MultiplyMatrixByVector(a, direction, size);
    double direction_dot_matrix_times_direction = Dot(direction, matrix_times_direction);
    double alpha = residual_norm_squared / direction_dot_matrix_times_direction;
    std::transform(solution.begin(), solution.end(), solution.begin(),
                   [&alpha, &direction](const double& val) { return val + (alpha * direction[0]); });
    std::transform(
        residual.begin(), residual.end(), residual.begin(),
        [&alpha, &matrix_times_direction](const double& val) { return val - (alpha * matrix_times_direction[0]); });

    double new_residual_norm_squared = Dot(residual);
    double beta = new_residual_norm_squared / residual_norm_squared;
    residual_norm_squared = new_residual_norm_squared;
    for (int i = 0; i < size; i++) {
      direction[i] = residual[i] + beta * direction[i];
    }
  }
  return solution;
}

inline double CalculateDeterminant(const double* a, int size) {
  if (size == 1) {
    return a[0];
  }
  if (size == 2) {
    return (a[0] * a[3]) - (a[1] * a[2]);
  }
  if (size > 2) {
    double det = 0;
    for (int i = 0; i < size; i++) {
      std::vector<std::vector<double>> submatrix(size - 1, std::vector<double>(size - 1));
      for (int j = 1; j < size; j++) {
        for (int k = 0; k < size; k++) {
          if (k != i) {
            submatrix[j - 1][k] = a[(j * size) + k];
          }
        }
      }

      det += std::pow(-1, i) * a[i] * CalculateDeterminant(a + (size * 1), size - 1);
    }
    return det;
  }
}

inline bool MatrixSimmPositive(const double* a, int size) {
  std::vector<double> a0(size * size);
  std::copy(a, a + (size * size), a0.begin());

  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (a0[(i * size) + j] != a0[(j * size) + i]) {
        return false;
      }
    }
  }

  std::vector<double> minors(size);
  for (int i = 1; i <= size; i++) {
    auto* submatrix = new double[i * i];
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < i; k++) {
        submatrix[(j * i) + k] = a[(j * size) + k];
      }
    }

    minors[i - 1] = CalculateDeterminant(submatrix, i);
    delete[] submatrix;
  }
  for (double minor : minors) {
    if (minor <= 0) {
      return false;
    }
  }
  return true;
}

class GradientMethod : public ppc::core::Task {
 public:
  explicit GradientMethod(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int size_;
  double tolerance_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> solution_;
  std::vector<double> result_;
};

}  // namespace sidorina_p_gradient_method_seq