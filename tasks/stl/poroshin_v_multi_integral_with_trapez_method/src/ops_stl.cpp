#include "stl/poroshin_v_multi_integral_with_trapez_method/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
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

void poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::CountMultiIntegralTrapezMethodStl() {
  const int dimensions = static_cast<int>(limits_.size());
  std::vector<double> h(dimensions);

  const int num_threads = ppc::util::GetPPCNumThreads();

  auto parallel_for = [num_threads](int start, int end, auto &&func) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    const int chunk_size = (end - start + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
      int thread_start = start + i * chunk_size;
      int thread_end = std::min(thread_start + chunk_size, end);

      if (thread_start < thread_end) {
        threads.emplace_back([=, &func]() {
          for (int j = thread_start; j < thread_end; ++j) {
            func(j);
          }
        });
      }
    }

    for (auto &thread : threads) {
      thread.join();
    }
  };

  parallel_for(0, dimensions, [&](int i) { h[i] = (limits_[i].second - limits_[i].first) / n_[i]; });

  std::vector<std::vector<double>> weights(dimensions);
  parallel_for(0, dimensions, [&](int i) {
    weights[i].resize(n_[i] + 1);
    for (int j = 0; j <= n_[i]; ++j) {
      weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
    }
  });

  int total_points = 1;
  for (int n : n_) {
    total_points *= (n + 1);
  }

  std::vector<std::pair<double, double>> thread_results(num_threads, {0.0, 1.0});
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const int chunk_size = (total_points + num_threads - 1) / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    int thread_start = i * chunk_size;
    int thread_end = std::min(thread_start + chunk_size, total_points);

    if (thread_start < thread_end) {
      threads.emplace_back([&, thread_start, thread_end]() {
        double local_integral = 0.0;
        double local_volume = 1.0;
        std::vector<double> vars(dimensions);
        std::vector<int> indices(dimensions, 0);

        for (int linear_idx = thread_start; linear_idx < thread_end; ++linear_idx) {
          int idx = linear_idx;
          for (int dim = dimensions - 1; dim >= 0; --dim) {
            indices[dim] = idx % (n_[dim] + 1);
            idx /= (n_[dim] + 1);
          }

          double weight = 1.0;
          for (int dim = 0; dim < dimensions; ++dim) {
            vars[dim] = limits_[dim].first + indices[dim] * h[dim];
            weight *= weights[dim][indices[dim]];
          }

          local_integral += func_(vars) * weight;
        }

        double chunk_fraction = static_cast<double>(thread_end - thread_start) / total_points;
        for (double hi : h) {
          local_volume *= hi;
        }
        local_volume *= chunk_fraction;

        thread_results[i] = {local_integral, local_volume};
      });
    }
  }

  for (auto &thread : threads) {
    thread.join();
  }

  double integral = 0.0;
  double volume = 0.0;
  for (const auto &result : thread_results) {
    integral += result.first;
    volume += result.second;
  }

  res_ = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::PreProcessingImpl() {
  n_.resize(dim_);
  limits_.resize(dim_);
  for (size_t i = 0; i < dim_; i++) {
    n_[i] = reinterpret_cast<int *>(task_data->inputs[0])[i];
    limits_[i].first = reinterpret_cast<double *>(task_data->inputs[1])[i];
    limits_[i].second = reinterpret_cast<double *>(task_data->inputs[2])[i];
  }
  res_ = 0;
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim_);
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::RunImpl() {
  CountMultiIntegralTrapezMethodStl();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_stl::TestTaskSTL::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
