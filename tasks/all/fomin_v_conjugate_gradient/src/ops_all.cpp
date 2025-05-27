#include "all/fomin_v_conjugate_gradient/include/ops_all.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

double fomin_v_conjugate_gradient::FominVConjugateGradientAll::DotProduct(const boost::mpi::communicator& world,
                                                                          const std::vector<double>& a,
                                                                          const std::vector<double>& b) {
  double local_sum = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, a.size()), 0.0,
      [&](const tbb::blocked_range<size_t>& r, double sum) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          sum += a[i] * b[i];
        }
        return sum;
      },
      std::plus<>());

  double global_sum;
  all_reduce(world, local_sum, global_sum, std::plus<>());
  return global_sum;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::MatrixVectorMultiply(
    const std::vector<double>& a, const std::vector<double>& x) const {
  std::vector<double> local_result(rows_per_proc, 0.0);

  tbb::parallel_for(tbb::blocked_range<int>(0, rows_per_proc), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += a[i * n + j] * x[j];
      }
      local_result[i] = sum;
    }
  });

  std::vector<double> global_result;
  gather(world_, local_result.data(), rows_per_proc, global_result, 0);
  broadcast(world_, global_result, 0);

  return global_result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorAdd(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorSub(const std::vector<double>& a,
                                                                                      const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

std::vector<double> fomin_v_conjugate_gradient::FominVConjugateGradientAll::VectorScalarMultiply(
    const std::vector<double>& v, double scalar) {
  std::vector<double> result(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = v[i] * scalar;
  }
  return result;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::vector<double> input(in_ptr, in_ptr + input_size);

    n = static_cast<int>((-1.0 + std::sqrt(1 + 4 * input_size)) / 2);
    a_ = std::vector<double>(input.begin(), input.begin() + (n * n));
    b_ = std::vector<double>(input.begin() + (n * n), input.end());
  }

  broadcast(world_, n, 0);
  rows_per_proc = n / world_.size();

  local_a_.resize(rows_per_proc * n);
  boost::mpi::scatter(world_, a_.data(), local_a_.data(), rows_per_proc * n, 0);

  local_b_.resize(rows_per_proc);
  boost::mpi::scatter(world_, b_.data(), local_b_.data(), rows_per_proc, 0);

  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  const int calculated_n = static_cast<int>((-1.0 + std::sqrt(1 + (4 * input_size))) / 2);
  return (static_cast<unsigned int>(calculated_n * (calculated_n + 1)) == input_size) &&
         (task_data->outputs_count[0] == static_cast<unsigned int>(calculated_n));
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::RunImpl() {
  std::vector<double> x(n, 0.0);
  std::vector<double> r = b_;
  std::vector<double> p = r;

  double rs_old = DotProduct(world_, r, r);

  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<double> ap = MatrixVectorMultiply(a_, p);
    double p_ap = DotProduct(world_, p, ap);

    if (world_.rank() == 0 && std::abs(p_ap) < 1e-12) break;
    broadcast(world_, p_ap, 0);

    double alpha = rs_old / p_ap;
    x = VectorAdd(x, VectorScalarMultiply(p, alpha));
    r = VectorSub(r, VectorScalarMultiply(ap, alpha));

    double rs_new = DotProduct(world_, r, r);
    if (world_.rank() == 0 && std::sqrt(rs_new) < epsilon) break;
    broadcast(world_, rs_new, 0);

    double beta = rs_new / rs_old;
    p = VectorAdd(r, VectorScalarMultiply(p, beta));
    rs_old = rs_new;
  }

  if (world_.rank() != 0) x.resize(n);
  gather(world_, x.data(), rows_per_proc, x.data(), 0);
  output_ = x;
  return true;
}

bool fomin_v_conjugate_gradient::FominVConjugateGradientAll::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}
