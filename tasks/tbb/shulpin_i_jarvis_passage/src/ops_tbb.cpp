#include "tbb/shulpin_i_jarvis_passage/include/ops_tbb.hpp"

#include <oneapi/tbb/concurrent_unordered_set.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
int Orientation(const shulpin_i_jarvis_tbb::Point& p, const shulpin_i_jarvis_tbb::Point& q,
                const shulpin_i_jarvis_tbb::Point& r) {
  int val = static_cast<int>(((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y)));
  if (val == 0) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}
}  // namespace

void shulpin_i_jarvis_tbb::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_tbb::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_tbb::Point>& output_jar) {
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

bool shulpin_i_jarvis_tbb::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_tbb::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_tbb::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_tbb::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_tbb::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_tbb::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

void shulpin_i_jarvis_tbb::JarvisTBBParallel::MakeJarvisPassageTBB(
    std::vector<shulpin_i_jarvis_tbb::Point>& input_jar, std::vector<shulpin_i_jarvis_tbb::Point>& output_jar) {
  size_t total_size_t = input_jar.size();
  auto total = static_cast<int32_t>(total_size_t);
  output_jar.clear();

  if (total < 3) return;

  int32_t start = 0;
  for (int32_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  int32_t active = start;
  std::vector<shulpin_i_jarvis_tbb::Point> hull;
  tbb::concurrent_unordered_set<shulpin_i_jarvis_tbb::Point, PointHash> unique_points;
  int32_t max_iterations = total;

  do {
    if (unique_points.insert(input_jar[active]).second) {
      hull.push_back(input_jar[active]);
    }

    int32_t candidate_init = (active + 1) % total;
    std::atomic<int32_t> candidate(candidate_init);
    tbb::enumerable_thread_specific<int32_t> local_candidates(candidate_init);

    tbb::parallel_for(tbb::blocked_range<int32_t>(0, total), [&](const tbb::blocked_range<int32_t>& range) {
      int32_t local = candidate;
      for (int32_t index = range.begin(); index < range.end(); ++index) {
        if (Orientation(input_jar[active], input_jar[index], input_jar[local]) == 2) {
          local = index;
        }
      }
      local_candidates.local() = local;
    });

    int32_t best_candidate = candidate;
    for (const auto& local : local_candidates) {
      if (Orientation(input_jar[active], input_jar[local], input_jar[best_candidate]) == 2) {
        best_candidate = local;
      }
    }

    if (best_candidate == active || max_iterations-- <= 0) {
      break;
    }
    active = best_candidate;

  } while (active != start);

  output_jar = std::move(hull);
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_tbb::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_tbb::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_tbb_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_tbb_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::RunImpl() {
  MakeJarvisPassageTBB(input_tbb_, output_tbb_);
  return true;
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_tbb_.begin(), output_tbb_.end(), result);
  return true;
}