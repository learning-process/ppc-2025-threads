#include "stl/fomin_v_conjugate_gradient/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

double fomin_v_conjugate_gradient::FominVConjugateGradientStl::DotProduct(const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::MatrixVectorMultiply(
    const std::vector<double>& a, const std::vector<double>& x) const {
  std::vector<double> result(n, 0.0);
  const auto num_threads = static_cast<size_t>(std::thread::hardware_concurrency());
  std::vector<std::thread> threads(num_threads);

  auto worker = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      result[i] = std::inner_product(a.begin() + i * n, a.begin() + (i + 1) * n, x.begin(), 0.0);
    }
  };

  const size_t chunk = static_cast<size_t>(n) / num_threads;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t start = t * chunk;
    size_t end = (t == num_threads - 1) ? static_cast<size_t>(n) : (t + 1) * chunk;
    threads[t] = std::thread(worker, start, end);
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>());
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<>());
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientStl::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  std::transform(v.begin(), v.end(), result.begin(), [scalar](double val) { return val * scalar; });
  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::vector<double> input(in_ptr, in_ptr + input_size);

  n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
  b_ = std::vector<double>(input.begin() + (n * n), input.end());
  output_.resize(n, 0.0);

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::RunImpl() {
  const double epsilon = 1e-6;
  const int max_iter = 1000;
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;
  double rs_old = DotProduct(r, r);

  for (int i = 0; i < max_iter; ++i) {
    std::vector<double> ap = MatrixVectorMultiply(a_, p);
    double p_ap = DotProduct(p, ap);

    if (std::abs(p_ap) < 1e-12) {
      break;
    }

    double alpha = rs_old / p_ap;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    std::vector<double> r_new = VectorSub(r, VectorScalarMultiply(ap, alpha));

    double rs_new = DotProduct(r_new, r_new);
    if (std::sqrt(rs_new) < epsilon) {
      break;
    }

    double beta = rs_new / rs_old;
    p = VectorAdd(r_new, VectorScalarMultiply(p, beta));
    r = r_new;
    rs_old = rs_new;
  }

  output_ = x;
  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientStl::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
