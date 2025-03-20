#include "omp/shulpin_i_jarvis_passage/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  int* result = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(reinterpret_cast<int*>(output_seq_.data()),
                    reinterpret_cast<int*>(output_seq_.data() + output_seq_.size()), result);
  return true;
}

void shulpin_i_jarvis_omp::JarvisOMPParallel::MakeJarvisPassageOMP(
    std::vector<shulpin_i_jarvis_omp::Point>& input_jar, std::vector<shulpin_i_jarvis_omp::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  size_t start = 0;
#pragma omp parallel
  {
    size_t local_start = start;

#pragma omp for nowait
    for (int i = 1; i < static_cast<int>(total); ++i) {
      if (input_jar[i].x < input_jar[local_start].x ||
          (input_jar[i].x == input_jar[local_start].x && input_jar[i].y < input_jar[local_start].y)) {
        local_start = i;
      }
    }

#pragma omp critical
    {
      if (input_jar[local_start].x < input_jar[start].x ||
          (input_jar[local_start].x == input_jar[start].x && input_jar[local_start].y < input_jar[start].y)) {
        start = local_start;
      }
    }
  }

  size_t active = start;

  do {
    output_jar.push_back(input_jar[active]);
    size_t candidate = (active + 1) % total;

#pragma omp parallel for shared(candidate)
    for (int index = 0; index < static_cast<int>(total); ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
#pragma omp critical
        { candidate = static_cast<size_t>(index); }
      }
    }
    active = candidate;
  } while (active != start);
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
  int* result = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(reinterpret_cast<int*>(output_omp_.data()),
                    reinterpret_cast<int*>(output_omp_.data() + output_omp_.size()), result);
  return true;
}
