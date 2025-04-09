#include "stl/poroshin_v_multi_integral_with_trapez_method/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

// void poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::CountMultiIntegralTrapezMethodStl() {
//   const int dimensions = static_cast<int>(limits_.size());
//   std::vector<double> h(dimensions);
//
// #pragma omp parallel for schedule(static)
//   for (int i = 0; i < dimensions; ++i) {
//     h[i] = (limits_[i].second - limits_[i].first) / n_[i];
//   }
//
//   std::vector<std::vector<double>> weights(dimensions);
// #pragma omp parallel for schedule(static)
//   for (int i = 0; i < dimensions; ++i) {
//     weights[i].resize(n_[i] + 1);
//     for (int j = 0; j <= n_[i]; ++j) {
//       weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
//     }
//   }
//
//   double integral = 0.0;
//
// #pragma omp parallel reduction(+ : integral)
//   {
//     std::vector<double> vars(dimensions);
//     std::vector<int> indices(dimensions, 0);
//
//     int total_points = 1;
//     for (int n : n_) {
//       total_points *= (n + 1);
//     }
//
// #pragma omp for schedule(static)
//     for (int linear_idx = 0; linear_idx < total_points; ++linear_idx) {
//       int idx = linear_idx;
//       for (int dim = dimensions - 1; dim >= 0; --dim) {
//         indices[dim] = idx % (n_[dim] + 1);
//         idx /= (n_[dim] + 1);
//       }
//
//       double weight = 1.0;
//       for (int dim = 0; dim < dimensions; ++dim) {
//         vars[dim] = limits_[dim].first + indices[dim] * h[dim];
//         weight *= weights[dim][indices[dim]];
//       }
//
//       integral += func_(vars) * weight;
//     }
//   }
//
//   double volume = 1.0;
// #pragma omp parallel for reduction(* : volume)
//   for (int i = 0; i < dimensions; ++i) {
//     volume *= h[i];
//   }
//
//   res_ = integral * volume;
// }

namespace {

void ParallelFor(int start, int end, int num_threads, std::function<void(int)> func) {
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const int chunk_size = (end - start + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    int thread_start = start + (i * chunk_size);
    int thread_end = std::min(thread_start + chunk_size, end);

    if (thread_start < thread_end) {
      threads.emplace_back([=, &func]() {
        for (int j = thread_start; j < thread_end; ++j) {
          func(j);
        }
      });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace

void poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::CountMultiIntegralTrapezMethodStl() {
  const int dimensions = static_cast<int>(limits_.size());
  const int num_threads = ppc::util::GetPPCNumThreads();

  std::vector<double> h(dimensions);
  ParallelFor(0, dimensions, num_threads, [&](int i) { h[i] = (limits_[i].second - limits_[i].first) / n_[i]; });

  std::vector<std::vector<double>> weights(dimensions);
  ParallelFor(0, dimensions, num_threads, [&](int i) {
    weights[i].resize(n_[i] + 1);
    for (int j = 0; j <= n_[i]; ++j) {
      weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
    }
  });

  int total_points = 1;
  for (int n : n_) {
    total_points *= (n + 1);
  }

  double volume = std::accumulate(h.begin(), h.end(), 1.0, std::multiplies<>());

  std::vector<double> thread_integrals(num_threads, 0.0);
  ParallelFor(0, total_points, num_threads, [&](int linear_idx) {
    std::vector<double> vars(dimensions);
    std::vector<int> indices(dimensions, 0);

    int idx = linear_idx;
    for (int dim = dimensions - 1; dim >= 0; --dim) {
      indices[dim] = idx % (n_[dim] + 1);
      idx /= (n_[dim] + 1);
    }

    double weight = 1.0;
    for (int dim = 0; dim < dimensions; ++dim) {
      vars[dim] = limits_[dim].first + (indices[dim] * h[dim]);
      weight *= weights[dim][indices[dim]];
    }

    thread_integrals[linear_idx % num_threads] += func_(vars) * weight;
  });

  double integral = std::accumulate(thread_integrals.begin(), thread_integrals.end(), 0.0);
  res_ = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim_);
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::RunImpl() {
  CountMultiIntegralTrapezMethodStl();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;
  return true;
}
