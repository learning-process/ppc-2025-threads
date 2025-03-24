#include "stl/lopatin_i_monte_carlo/include/lopatinMonteCarloSTL.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace lopatin_i_monte_carlo_stl {

bool TestTaskSTL::ValidationImpl() {
  const bool outputs_valid = !task_data->outputs_count.empty() && task_data->outputs_count[0] == 1;
  const bool inputs_valid = task_data->inputs_count.size() == 2 &&
                            (task_data->inputs_count[0] % 2 == 0) &&  // odd num of bounds
                            task_data->inputs_count[1] == 1;          // iterations num

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  const int iterations = *iter_ptr;
  const bool iter_valid = iterations > 0;

  return outputs_valid && inputs_valid && iter_valid;
}

bool TestTaskSTL::PreProcessingImpl() {
  auto* bounds_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t bounds_size = task_data->inputs_count[0];
  integrationBounds_.resize(bounds_size);
  std::copy(bounds_ptr, bounds_ptr + bounds_size, integrationBounds_.begin());

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  iterations_ = *iter_ptr;
  return true;
}

bool TestTaskSTL::RunImpl() {
  const size_t d = integrationBounds_.size() / 2;  // dimensions

  const int numThreads = ppc::util::GetPPCNumThreads();
  // printf("%d\n", numThreads);
  std::vector<std::thread> threads(numThreads);
  std::vector<double> partialSums(numThreads, 0.0);
  const int chunkSize = iterations_ / numThreads;
  const int remainder = iterations_ % numThreads;

  // init random numbers generator
  std::vector<std::mt19937> generators;
  std::random_device rd;
  std::seed_seq seed{rd(), static_cast<unsigned int>(std::time(nullptr))};
  std::vector<std::mt19937::result_type> seeds(numThreads);
  seed.generate(seeds.begin(), seeds.end());
  for (auto& s : seeds) {
    generators.emplace_back(s);
  }

  // volume of integration region
  double volume = 1.0;
  for (size_t j = 0; j < d; ++j) {
    volume *= (integrationBounds_[(2 * j) + 1] - integrationBounds_[2 * j]);
  }

  auto thread_task = [&](int thread_id, int start, int end) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto& gen = generators[thread_id];
    double local_sum = 0.0;

    for (int i = start; i < end; ++i) {
      std::vector<double> point(d);
      for (size_t j = 0; j < d; ++j) {
        const double min = integrationBounds_[2 * j];
        const double max = integrationBounds_[(2 * j) + 1];
        point[j] = min + (max - min) * dis(gen);
      }
      local_sum += integrand_(point);
    }

    partialSums[thread_id] = local_sum;
  };

  // create and run threads
  int start = 0;
  for (int tid = 0; tid < numThreads; ++tid) {
    const int end = start + chunkSize + (tid < remainder ? 1 : 0);
    threads[tid] = std::thread(thread_task, tid, start, end);
    start = end;
  }

  // waiting for all threads to end their work
  for (auto& t : threads) {
    t.join();
  }

  double total_sum = std::accumulate(partialSums.begin(), partialSums.end(), 0.0);

  result_ = (total_sum / iterations_) * volume;

  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result_;
  return true;
}

}  // namespace lopatin_i_monte_carlo_stl
