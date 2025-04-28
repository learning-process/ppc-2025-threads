#include "stl/shulpin_i_jarvis_passage/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_set>

#include "core/util/include/util.hpp"

namespace {
int Orientation(const shulpin_i_jarvis_stl::Point& p, const shulpin_i_jarvis_stl::Point& q,
                const shulpin_i_jarvis_stl::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}
}  // namespace

void shulpin_i_jarvis_stl::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_stl::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_stl::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    output_jar.emplace_back(input_jar[active]);
    size_t candidate = (active + 1) % total;

    for (size_t index = 0; index < total; ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;
  } while (active != start);
}

bool shulpin_i_jarvis_stl::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_stl::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_stl::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_stl::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_stl::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_stl::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

void shulpin_i_jarvis_stl::JarvisSTLParallel::MakeJarvisPassageSTL(
    std::vector<shulpin_i_jarvis_stl::Point>& input_jar, std::vector<shulpin_i_jarvis_stl::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  std::unordered_set<shulpin_i_jarvis_stl::Point, shulpin_i_jarvis_stl::PointHash, shulpin_i_jarvis_stl::PointEqual>
      unique_points;

  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    const auto& current = input_jar[active];
    if (unique_points.find(current) == unique_points.end()) {
      output_jar.emplace_back(current);
      unique_points.insert(current);
    }

    size_t candidate = (active + 1) % total;

    const unsigned num_threads = ppc::util::GetPPCNumThreads();
    std::vector<std::thread> threads;
    std::vector<size_t> local_candidates(num_threads, candidate);

    auto worker = [&](unsigned t) {
      size_t from = t * total / num_threads;
      size_t to = (t + 1 == num_threads) ? total : (t + 1) * total / num_threads;
      size_t best = candidate;

      for (size_t i = from; i < to; ++i) {
        if (i == active) {
          continue;
        }
        if (Orientation(current, input_jar[i], input_jar[best]) == 2) {
          best = i;
        }
      }

      local_candidates[t] = best;
    };

    threads.reserve(num_threads);
    for (unsigned t = 0; t < num_threads; ++t) {
      threads.emplace_back(worker, t);
    }
    for (auto& th : threads) {
      th.join();
    }

    for (size_t i = 0; i < num_threads; ++i) {
      if (Orientation(current, input_jar[local_candidates[i]], input_jar[candidate]) == 2) {
        candidate = local_candidates[i];
      }
    }

    active = candidate;

  } while (active != start);
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_stl::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_stl::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_stl_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_stl_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::RunImpl() {
  MakeJarvisPassageSTL(input_stl_, output_stl_);
  return true;
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_stl_.begin(), output_stl_.end(), result);
  return true;
}