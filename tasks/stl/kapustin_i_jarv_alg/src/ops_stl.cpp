#include "stl/kapustin_i_jarv_alg/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

int kapustin_i_jarv_alg_stl::TestTaskSTL::CalculateDistance(const std::pair<int, int>& p1,
                                                            const std::pair<int, int>& p2) {
  return static_cast<int>(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

int kapustin_i_jarv_alg_stl::TestTaskSTL::Orientation(const std::pair<int, int>& p, const std::pair<int, int>& q,
                                                      const std::pair<int, int>& r) {
  int val = ((q.second - p.second) * (r.first - q.first)) - ((q.first - p.first) * (r.second - q.second));
  if (val == 0) {
    return 0;
  }
  return (val > 0) ? 1 : -1;
}

bool kapustin_i_jarv_alg_stl::TestTaskSTL::PreProcessingImpl() {
  std::vector<std::pair<int, int>> points;

  for (size_t i = 0; i < task_data->inputs.size(); ++i) {
    auto* data = reinterpret_cast<std::pair<int, int>*>(task_data->inputs[i]);
    size_t count = task_data->inputs_count[i];
    points.assign(data, data + count);
  }
  input_ = points;

  leftmost_index_ = 0;
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i].first < input_[leftmost_index_].first) {
      leftmost_index_ = i;
    }
  }

  current_point_ = input_[leftmost_index_];

  return true;
}

bool kapustin_i_jarv_alg_stl::TestTaskSTL::ValidationImpl() { return !task_data->inputs.empty(); }

bool kapustin_i_jarv_alg_stl::TestTaskSTL::RunImpl() {
  std::pair<int, int> start_point = current_point_;
  size_t current_index = leftmost_index_;
  output_.clear();
  output_.push_back(start_point);

  const size_t num_threads = static_cast<size_t>(ppc::util::GetPPCNumThreads());
  const size_t chunk_size = (input_.size() + num_threads - 1) / num_threads;

  do {
    size_t best_index = (current_index + 1) % input_.size();
    std::mutex mutex;

    auto worker = [&](size_t start, size_t end) {
      size_t local_best = best_index;
      for (size_t i = start; i < end; ++i) {
        if (i == current_index) continue;

        int orientation = Orientation(input_[current_index], input_[local_best], input_[i]);
        if (orientation > 0 || (orientation == 0 && CalculateDistance(input_[i], input_[current_index]) >
                                                        CalculateDistance(input_[local_best], input_[current_index]))) {
          local_best = i;
        }
      }

      std::lock_guard<std::mutex> lock(mutex);
      int orientation = Orientation(input_[current_index], input_[best_index], input_[local_best]);
      if (orientation > 0 || (orientation == 0 && CalculateDistance(input_[local_best], input_[current_index]) >
                                                      CalculateDistance(input_[best_index], input_[current_index]))) {
        best_index = local_best;
      }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
      size_t start = i * chunk_size;
      size_t end = std::min(start + chunk_size, input_.size());
      threads.emplace_back(worker, start, end);
    }

    for (auto& t : threads) {
      t.join();
    }

    if (!output_.empty() && input_[best_index] == output_.front()) {
      break;
    }

    current_point_ = input_[best_index];
    output_.push_back(current_point_);
    current_index = best_index;

  } while (current_point_ != start_point);

  return true;
}

bool kapustin_i_jarv_alg_stl::TestTaskSTL::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<std::pair<int, int>*>(task_data->outputs[0]);
  std::ranges::copy(output_, result_ptr);
  return true;
}