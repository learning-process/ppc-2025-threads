#pragma once

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <stdexcept>  // Required for std::runtime_error
#include <string>     // Required for std::string
#include <thread>     // Required for std::thread
#include <tuple>      // Required for std::tie
#include <utility>    // Required for std::move
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"  // For GetPPCNumThreads

namespace alputov_i_graham_scan_all {

struct Point {
  double x, y;
  Point(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
  bool operator<(const Point& other) const {
    // Standard sort: by Y, then by X
    if (y != other.y) {
      return y < other.y;
    }
    return x < other.x;
  }
  bool operator==(const Point& other) const {
    constexpr double kEpsilon = 1e-9;
    return std::abs(x - other.x) < kEpsilon && std::abs(y - other.y) < kEpsilon;
  }
  bool operator!=(const Point& other) const { return !(*this == other); }
};

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data);
  ~TestTaskALL() override;

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // MPI specific
  int rank_{};
  int world_size_{};
  MPI_Datatype mpi_point_datatype_{};
  MPI_Comm active_comm_ = MPI_COMM_NULL;  // Communicator for active processes
  int active_procs_count_ = 0;

  // Data
  std::vector<Point> input_points_;            // On rank 0: all input points
  std::vector<Point> convex_hull_;             // On rank 0: final result
  Point pivot_;                                // Pivot point, available on all active ranks after broadcast
  std::vector<Point> local_points_;            // Points local to each process for sorting
  std::vector<Point> globally_sorted_points_;  // On rank 0: all points sorted globally

  // Core Graham Scan logic (adapted from alputov_i_graham_scan_stl)
  static Point FindPivot(const std::vector<Point>& points);
  static bool CompareAngles(const Point& p1, const Point& p2, const Point& pivot);
  static double CrossProduct(const Point& o, const Point& a, const Point& b);
  static void RemoveDuplicates(std::vector<Point>& points);
  static std::vector<Point> BuildHull(const std::vector<Point>& sorted_points, const Point& pivot);
  static void LocalParallelSort(std::vector<Point>& points, const Point& pivot_for_sort);
};

}  // namespace alputov_i_graham_scan_all