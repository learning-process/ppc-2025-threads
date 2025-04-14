#include "tbb/alputov_i_graham_scan/include/ops_tbb.hpp"

#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>

namespace alputov_i_graham_scan_tbb {

Point::Point(double x, double y) : x(x), y(y) {}

bool Point::operator<(const Point& other) const { return std::tie(y, x) < std::tie(other.y, other.x); }

bool Point::operator==(const Point& other) const { return std::tie(x, y) == std::tie(other.x, other.y); }

bool TestTaskTBB::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskTBB::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs_count[0] <= task_data->outputs_count[0]);
}

double TestTaskTBB::Cross(const Point& o, const Point& a, const Point& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

Point TestTaskTBB::FindPivot() const {
  auto min_it = tbb::parallel_reduce(
      tbb::blocked_range(input_points_.begin(), input_points_.end()), input_points_.begin(),
      [](const auto& r, auto curr_min) {
        auto local_min = std::min_element(r.begin(), r.end());
        return (*local_min < *curr_min) ? local_min : curr_min;
      },
      [](auto a, auto b) { return (*a < *b) ? a : b; });
  return *min_it;
}

void TestTaskTBB::RemoveDuplicates(std::vector<Point>& points) const {
  auto last = std::unique(points.begin(), points.end());
  points.erase(last, points.end());
}

bool TestTaskTBB::CompareAngles(const Point& a, const Point& b, const Point& pivot) const {
  const auto dx1 = a.x - pivot.x;
  const auto dy1 = a.y - pivot.y;
  const auto dx2 = b.x - pivot.x;
  const auto dy2 = b.y - pivot.y;

  const auto cross = dx1 * dy2 - dy1 * dx2;
  return cross != 0 ? cross > 0 : (dx1 * dx1 + dy1 * dy1) < (dx2 * dx2 + dy2 * dy2);
}

std::vector<Point> TestTaskTBB::SortPoints(const Point& pivot) const {
  std::vector<Point> points;
  points.reserve(input_points_.size() - 1);
  for (const auto& p : input_points_) {
    if (p != pivot) points.push_back(p);
  }

  tbb::parallel_sort(points, [&](const Point& a, const Point& b) { return CompareAngles(a, b, pivot); });
  RemoveDuplicates(points);
  return points;
}

std::vector<Point> TestTaskTBB::BuildHull(const std::vector<Point>& sorted_points, const Point& pivot) const {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size() + 1);
  hull.push_back(pivot);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  return hull;
}

bool TestTaskTBB::RunImpl() {
  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);

  if (sorted_points.empty()) {
    convex_hull_ = {pivot};
    return true;
  }

  tbb::combinable<std::vector<Point>> local_hulls([&] {
    std::vector<Point> hull;
    hull.reserve(sorted_points.size() / tbb::this_task_arena::max_concurrency());
    hull.push_back(pivot);
    return hull;
  });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_points.size()), [&](const tbb::blocked_range<size_t>& range) {
    auto& hull = local_hulls.local();
    for (size_t i = range.begin(); i < range.end(); ++i) {
      const auto& p = sorted_points[i];
      while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
        hull.pop_back();
      }
      hull.push_back(p);
    }
  });

  std::vector<Point> combined;
  local_hulls.combine_each(
      [&](const std::vector<Point>& hull) { combined.insert(combined.end(), hull.begin(), hull.end()); });

  combined.erase(std::remove(combined.begin(), combined.end(), pivot), combined.end());
  combined.insert(combined.begin(), pivot);
  RemoveDuplicates(combined);

  convex_hull_ = BuildHull(std::vector<Point>(combined.begin() + 1, combined.end()), pivot);
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::copy(convex_hull_.begin(), convex_hull_.end(), output_ptr);
  return true;
}

const std::vector<Point>& TestTaskTBB::GetConvexHull() const { return convex_hull_; }

}  // namespace alputov_i_graham_scan_tbb