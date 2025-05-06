#include "stl/alputov_i_graham_scan/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace alputov_i_graham_scan_stl {

Point::Point(double x, double y) : x(x), y(y) {}

bool Point::operator<(const Point& other) const { return std::tie(y, x) < std::tie(other.y, other.x); }

bool Point::operator==(const Point& other) const {
  constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
  return std::abs(x - other.x) < kEpsilon && std::abs(y - other.y) < kEpsilon;
}

bool TestTaskSTL::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs_count[0] <= task_data->outputs_count[0]);
}

double TestTaskSTL::Cross(const Point& o, const Point& a, const Point& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

Point TestTaskSTL::FindPivot() const {
  const size_t num_threads = static_cast<size_t>(ppc::util::GetPPCNumThreads());
  const size_t num_points = input_points_.size();
  std::vector<std::thread> threads;
  std::vector<Point> partial_minima(num_threads);
  size_t points_per_thread = num_points / num_threads;
  size_t remainder = num_points % num_threads;

  for (size_t i = 0; i < num_threads; ++i) {
    size_t start = i * points_per_thread + std::min(i, remainder);
    size_t end = (i + 1) * points_per_thread + std::min(i + 1, remainder);

    threads.emplace_back([&, i, start, end]() {
      Point local_min = input_points_[start];
      for (size_t j = start + 1; j < end; ++j) {
        if (input_points_[j] < local_min) {
          local_min = input_points_[j];
        }
      }
      partial_minima[i] = local_min;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  Point global_min = partial_minima[0];
  for (size_t i = 1; i < num_threads; ++i) {
    if (partial_minima[i] < global_min) {
      global_min = partial_minima[i];
    }
  }

  return global_min;
}

bool TestTaskSTL::CompareAngles(const Point& first_point, const Point& second_point, const Point& pivot_point) {
  const auto first_dx = first_point.x - pivot_point.x;
  const auto first_dy = first_point.y - pivot_point.y;
  const auto second_dx = second_point.x - pivot_point.x;
  const auto second_dy = second_point.y - pivot_point.y;

  const double cross_product = (first_dx * second_dy) - (first_dy * second_dx);
  constexpr double kEpsilon = 1e-10;

  if (std::abs(cross_product) < kEpsilon) {
    const auto first_distance_squared = (first_dx * first_dx) + (first_dy * first_dy);
    const auto second_distance_squared = (second_dx * second_dx) + (second_dy * second_dy);
    return first_distance_squared < second_distance_squared;
  }

  return cross_product > 0;
}

void TestTaskSTL::RemoveDuplicates(std::vector<Point>& points) {
  auto result = std::unique(points.begin(), points.end());
  points.erase(result, points.end());
}

void TestTaskSTL::ParallelSort(std::vector<Point>& points, const Point& pivot) const {
  const size_t num_threads = static_cast<size_t>(ppc::util::GetPPCNumThreads());
  const size_t num_points = points.size();

  if (num_threads == 1 || num_points < num_threads * 2) {
    std::sort(points.begin(), points.end(), [&](const Point& a, const Point& b) { return CompareAngles(a, b, pivot); });
    return;
  }

  std::vector<std::thread> threads;
  size_t points_per_thread = num_points / num_threads;
  size_t remainder = num_points % num_threads;

  for (size_t i = 0; i < num_threads; ++i) {
    size_t start = i * points_per_thread + std::min(i, remainder);
    size_t end = (i + 1) * points_per_thread + std::min(i + 1, remainder);

    threads.emplace_back([&, start, end]() {
      std::sort(points.begin() + start, points.begin() + end,
                [&](const Point& a, const Point& b) { return CompareAngles(a, b, pivot); });
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::vector<Point> temp(num_points);
  for (size_t size = points_per_thread; size < num_points; size *= 2) {
    for (size_t left = 0; left < num_points; left += 2 * size) {
      size_t mid = std::min(left + size, num_points);
      size_t right = std::min(left + 2 * size, num_points);

      std::merge(points.begin() + left, points.begin() + mid, points.begin() + mid, points.begin() + right,
                 temp.begin() + left, [&](const Point& a, const Point& b) { return CompareAngles(a, b, pivot); });
    }
    points.swap(temp);
  }
}

std::vector<Point> TestTaskSTL::SortPoints(const Point& pivot) const {
  std::vector<Point> points;
  points.reserve(input_points_.size());
  for (const auto& p : input_points_) {
    if (!(p == pivot)) {
      points.push_back(p);
    }
  }

  ParallelSort(points, pivot);
  RemoveDuplicates(points);
  return points;
}

std::vector<Point> TestTaskSTL::BuildHull(const std::vector<Point>& sorted_points, const Point& pivot) const {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size() + 1);
  hull.push_back(pivot);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2) {
      const double cross = Cross(hull[hull.size() - 2], hull.back(), p);
      if (cross < 1e-8) {
        hull.pop_back();
      } else {
        break;
      }
    }
    hull.push_back(p);
  }

  while (hull.size() >= 3) {
    const double cross = Cross(hull[hull.size() - 2], hull.back(), hull[0]);
    if (cross < 1e-8) {
      hull.pop_back();
    } else {
      break;
    }
  }

  return hull;
}

bool TestTaskSTL::RunImpl() {
  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);

  if (sorted_points.empty()) {
    convex_hull_ = {pivot};
    return true;
  }

  convex_hull_ = BuildHull(sorted_points, pivot);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::copy(convex_hull_.begin(), convex_hull_.end(), output_ptr);
  return true;
}

const std::vector<Point>& TestTaskSTL::GetConvexHull() const { return convex_hull_; }

}  // namespace alputov_i_graham_scan_stl