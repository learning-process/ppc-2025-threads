#pragma once

#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sidorina_p_gradient_method_seq {

inline std::vector<double> MultiplyMatrixByVector(const std::vector<double>& a, const std::vector<double>& vec,
                                                  int size) {
  std::vector<double> result(size, 0);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i] += a[i * size + j] * vec[j];
    }
  }
  return result;
}

inline double VectorNorm(const std::vector<double>& vec) {
  double sum = 0;
  for (double value : vec) {
    sum += pow(value, 2);
  }
  return sqrt(sum);
}

inline double Dot(const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double sum = 0;
  for (size_t i = 0; i < vec1.size(); i++) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
}

inline double Dot(const std::vector<double>& vec) {
  double sum = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    sum += pow(vec[i], 2);
  }
  return sum;
}

inline std::vector<double> ConjugateGradientMethod(std::vector<double>& a, std::vector<double>& b,
                                                   std::vector<double> solution, double tolerance, int size) {
  std::vector<double> matrixTimesSolution = MultiplyMatrixByVector(a, solution, size);

  auto residual = std::vector<double>(size);
  auto direction = std::vector<double>(size);

std::transform(
      b.begin(), b.end(), residual.begin(),
      [&matrixTimesSolution](const double& val) { return val - matrixTimesSolution[0]; });


  double residualNormSquared = Dot(residual);
  if (sqrt(residualNormSquared) < tolerance) {
    return solution;
  }
  direction = residual;
  std::vector<double> matrixTimesDirection(size);
  while (sqrt(residualNormSquared) > tolerance) {
    matrixTimesDirection = MultiplyMatrixByVector(a, direction, size);
    double directionDotMatrixTimesDirection = Dot(direction, matrixTimesDirection);
    double alpha = residualNormSquared / directionDotMatrixTimesDirection;
    std::transform(solution.begin(), solution.end(), solution.begin(),
                   [&alpha, &direction](const double& val) { return val + alpha * direction[0]; });
    std::transform(
        residual.begin(), residual.end(), residual.begin(),
        [&alpha, &matrixTimesDirection](const double& val) { return val - alpha * matrixTimesDirection[0]; });

    double newResidualNormSquared = Dot(residual);
    double beta = newResidualNormSquared / residualNormSquared;
    residualNormSquared = newResidualNormSquared;
    for (int i = 0; i < size; i++) {
      direction[i] = residual[i] + beta * direction[i];
    }
  }
  return solution;
}

inline double calculateDeterminant(const double* a, int size) {
  if (size == 1) {
    return a[0];
  } else if (size == 2) {
    return a[0] * a[3] - a[1] * a[2];
  } else {
    double det = 0;
    for (int i = 0; i < size; i++) {
      std::vector<std::vector<double>> submatrix(size - 1, std::vector<double>(size - 1));
      for (int j = 1; j < size; j++) {
        for (int k = 0; k < size; k++) {
          if (k != i) {
            submatrix[j - 1][k] = a[j * size + k];
          }
        }
      }

      det += pow(-1, i) * a[i] * calculateDeterminant(a + (size * 1), size - 1);
    }
    return det;
  }
}

inline bool matrixSimmPositive(const double* a, int size) {
  std::vector<double> a0(size * size);
  std::copy(a, a + size * size, a0.begin()); 

  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (a0[i * size + j] != a0[j * size + i]) {
        return false;
      }
    }
  }

  std::vector<double> minors(size);
  for (int i = 1; i <= size; i++) {
    double* submatrix = new double[i * i]; 
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < i; k++) {
        submatrix[j * i + k] = a[j * size + k];
      }
    }

    minors[i - 1] = calculateDeterminant(submatrix, i); 
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
  int size;
  double tolerance;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> solution;
  std::vector<double> result;
};

}  // namespace sidorina_p_gradient_method_seq