#ifndef _INTEGRATOR_H_
#define _INTEGRATOR_H_
#include <cstdint>
#include <functional>
#include <stdexcept>
namespace khasanyanov_k_trapezoid_method_seq {

enum IntegrateTechnology : std::uint8_t { kSequential, kOpenMP, kTBB, kSTL, kMPI };

using IntegrateFunction = std::function<double(const std::vector<double>&)>;
using IntegrateBounds = std::pair<double, double>;

template <IntegrateTechnology technology>
class Integrator {
  static const int kDefaultParts;

  [[nodiscard]] static double calculate_weight(  // NOLINT(readability-identifier-naming)
      const std::vector<int>& indices, int steps);

  [[nodiscard]] static double trapezoidal_method(  // NOLINT(readability-identifier-naming)
      const IntegrateFunction& f, const std::vector<IntegrateBounds>& bounds, int steps);

  [[nodiscard]] static double trapezoidal_method_sequential(  // NOLINT(readability-identifier-naming)
      const IntegrateFunction&, const std::vector<IntegrateBounds>&, double, int, int);

 public:
  double operator()(const IntegrateFunction&, const std::vector<IntegrateBounds>&, double, int = kDefaultParts,
                    int = 1024) const;
};

//----------------------------------------------------------------------------------------------------------

template <IntegrateTechnology technology>
const int Integrator<technology>::kDefaultParts = 5;

template <IntegrateTechnology technology>
double Integrator<technology>::operator()(const IntegrateFunction& f, const std::vector<IntegrateBounds>& bounds,
                                          double precision,  // NOLINT(bugprone-easily-swappable-parameters)
                                          int init_steps, int max_steps) const {
  switch (technology) {
    case kSequential:
      return trapezoidal_method_sequential(f, bounds, precision, init_steps, max_steps);
    default:
      throw std::runtime_error("Technology not available");
  }
}

template <IntegrateTechnology technology>
double Integrator<technology>::trapezoidal_method_sequential(
    const IntegrateFunction& f, const std::vector<IntegrateBounds>& bounds,
    double precision,  // NOLINT(bugprone-easily-swappable-parameters)
    int init_steps, int max_steps) {
  int steps = init_steps;
  double prev_result = trapezoidal_method(f, bounds, steps);
  while (steps <= max_steps) {
    steps *= 2;
    double current_result = trapezoidal_method(f, bounds, steps);
    if (std::abs(current_result - prev_result) < precision) {
      return current_result;
    }
    prev_result = current_result;
  }
  return prev_result;
}

template <IntegrateTechnology technology>
double Integrator<technology>::trapezoidal_method(const IntegrateFunction& f,
                                                  const std::vector<IntegrateBounds>& bounds, int steps) {
  size_t dims = bounds.size();
  std::vector<double> dx(dims);

  for (size_t i = 0; i < dims; ++i) {
    dx[i] = (bounds[i].second - bounds[i].first) / steps;
  }

  double total = 0.0;
  std::vector<int> indices(dims, 0);
  bool done = false;

  while (!done) {
    std::vector<double> point(dims);
    for (size_t i = 0; i < dims; ++i) {
      point[i] = bounds[i].first + indices[i] * dx[i];
    }

    double weight = calculate_weight(indices, steps);
    total += weight * f(point);

    int j = 0;
    while (j < dims) {
      indices[j]++;
      if (indices[j] <= steps) {
        break;
      }

      indices[j] = 0;
      ++j;
    }
    if (j == dims) {
      done = true;
    }
  }

  double factor = 1.0;
  for (int i = 0; i < dims; ++i) {
    factor *= dx[i];
  }
  return total * factor;
}

template <IntegrateTechnology technology>
double Integrator<technology>::calculate_weight(const std::vector<int>& indices, int steps) {
  double weight = 1.0;
  for (int idx : indices) {
    weight *= (idx == 0 || idx == steps) ? 0.5 : 1.0;
  }
  return weight;
}

}  // namespace khasanyanov_k_trapezoid_method_seq

#endif