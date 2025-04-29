#include "stl/shulpin_i_jarvis_passage/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

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
  output_jar.clear();

  std::unordered_set<shulpin_i_jarvis_stl::Point, shulpin_i_jarvis_stl::PointHash, shulpin_i_jarvis_stl::PointEqual>
      unique_points;

  size_t most_left = 0;
  for (size_t i = 1; i < input_jar.size(); ++i) {
    if (input_jar[i].x < input_jar[most_left].x ||
        (input_jar[i].x == input_jar[most_left].x && input_jar[i].y < input_jar[most_left].y)) {
      most_left = i;
    }
  }

  const Point& minPoint = input_jar[most_left];
  std::vector<Point> convexHull = {minPoint};
  Point prevPoint = minPoint;
  Point nextPoint;

  auto findNextPoint = [](const Point& currentPoint, const std::vector<Point>& points, int start, int end,
                          Point& candidate) {
    for (int i = start; i < end; ++i) {
      const auto& point = points[i];
      if (point == currentPoint) continue;
      double crossProduct = (point.y - currentPoint.y) * (candidate.x - currentPoint.x) -
                            (point.x - currentPoint.x) * (candidate.y - currentPoint.y);
      double distCurrentPoint = std::pow(point.x - currentPoint.x, 2) + std::pow(point.y - currentPoint.y, 2);
      double distCandidate = std::pow(candidate.x - currentPoint.x, 2) + std::pow(candidate.y - currentPoint.y, 2);
      if (crossProduct > 0 || (crossProduct == 0 && distCurrentPoint > distCandidate)) candidate = point;
    }
  };

  do {
    nextPoint = input_jar[0];
    int numThreads = ppc::util::GetPPCNumThreads();
    int chunkSize = input_jar.size() / numThreads;
    std::vector<std::thread> threads;
    std::vector<Point> candidates(numThreads, nextPoint);

    for (int i = 0; i < numThreads; ++i) {
      int start = i * chunkSize;
      int end = (i == numThreads - 1) ? input_jar.size() : (i + 1) * chunkSize;
      threads.emplace_back(findNextPoint, std::ref(prevPoint), std::cref(input_jar), start, end,
                           std::ref(candidates[i]));
    }

    for (auto& thread : threads) {
      if (thread.joinable()) thread.join();
    }

    for (const auto& candidate : candidates) {
      double crossProduct = (candidate.y - prevPoint.y) * (nextPoint.x - prevPoint.x) -
                            (candidate.x - prevPoint.x) * (nextPoint.y - prevPoint.y);
      double distPrevPoint = std::pow(candidate.x - prevPoint.x, 2) + std::pow(candidate.y - prevPoint.y, 2);
      double distNextPoint = std::pow(nextPoint.x - prevPoint.x, 2) + std::pow(nextPoint.y - prevPoint.y, 2);
      if (crossProduct > 0 || (crossProduct == 0 && distPrevPoint > distNextPoint)) nextPoint = candidate;
    }

    if (unique_points.find(nextPoint) == unique_points.end()) {
      output_jar.push_back(nextPoint);
      unique_points.insert(nextPoint);
    }

    prevPoint = nextPoint;

  } while (nextPoint != minPoint);
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