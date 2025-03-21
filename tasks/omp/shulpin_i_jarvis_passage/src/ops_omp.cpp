#include "omp/shulpin_i_jarvis_passage/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
int Orientation(const shulpin_i_jarvis_omp::Point& p, const shulpin_i_jarvis_omp::Point& q,
                const shulpin_i_jarvis_omp::Point& r) {
  int val = static_cast<int>(((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y)));
  if (val == 0) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}
}  // namespace

void shulpin_i_jarvis_omp::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_omp::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_omp::Point>& output_jar) {
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

bool shulpin_i_jarvis_omp::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_omp::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_omp::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_omp::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_omp::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_omp::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

void shulpin_i_jarvis_omp::JarvisOMPParallel::MakeJarvisPassageOMP(
    std::vector<shulpin_i_jarvis_omp::Point>& input_jar, std::vector<shulpin_i_jarvis_omp::Point>& output_jar) {
  size_t total_size_t = input_jar.size();
  auto total = static_cast<int32_t>(total_size_t);
  output_jar.clear();
  
  int32_t start = 0;
#pragma omp parallel for
  for (int32_t i = 0; i < total; ++i) {
    int32_t candidate = i;
    for (int32_t j = i + 1; j < total; ++j) {
      if (input_jar[j].x < input_jar[candidate].x ||
          (input_jar[j].x == input_jar[candidate].x && input_jar[j].y < input_jar[candidate].y)) {
        candidate = j;
      }
    }

#pragma omp critical
    {
      if (input_jar[candidate].x < input_jar[start].x ||
          (input_jar[candidate].x == input_jar[start].x && input_jar[candidate].y < input_jar[start].y)) {
        start = candidate;
      }
    }
  }

  int32_t active = start;
  int32_t candidate = 0;
  std::vector<shulpin_i_jarvis_omp::Point> hull;
  hull.reserve(total);

  do {
    hull.push_back(input_jar[active]);
    candidate = (active + 1) % total;

#pragma omp parallel for shared(candidate)
    for (int32_t index = 0; index < total; ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
#pragma omp critical
        { candidate = index; }
      }
    }
    active = candidate;
  } while (active != start);

  output_jar = std::move(hull);
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_omp::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_omp::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_omp_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_omp_.resize(output_size);
  return true;
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::RunImpl() {
  MakeJarvisPassageOMP(input_omp_, output_omp_);
  return true;
}

bool shulpin_i_jarvis_omp::JarvisOMPParallel::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_omp_.begin(), output_omp_.end(), result);
  return true;
}
