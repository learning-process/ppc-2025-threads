#include "all/alputov_i_graham_scan/include/ops_all.hpp"

#include <algorithm>
#include <chrono>  // For std::chrono (potentially, if used internally for GetPPCNumThreads or similar)
#include <cmath>
#include <cstddef>
#include <iterator>  // For std::next
#include <numeric>   // For std::partial_sum, std::iota
#include <ranges>    // For std::ranges
#include <stdexcept>
#include <vector>

namespace alputov_i_graham_scan_all {

TestTaskALL::TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  // Create MPI Datatype for Point
  int blocklengths[2] = {1, 1};
  MPI_Aint displacements[2];
  MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
  Point p_dummy;
  MPI_Get_address(&p_dummy.x, &displacements[0]);
  MPI_Get_address(&p_dummy.y, &displacements[1]);
  displacements[1] -= displacements[0];  // Make relative
  displacements[0] = 0;
  MPI_Type_create_struct(2, blocklengths, displacements, types, &mpi_point_datatype_);
  MPI_Type_commit(&mpi_point_datatype_);
}

TestTaskALL::~TestTaskALL() {
  if (mpi_point_datatype_ != MPI_DATATYPE_NULL) {
    MPI_Type_free(&mpi_point_datatype_);
  }
  if (active_comm_ != MPI_COMM_NULL && active_comm_ != MPI_COMM_WORLD) {
    MPI_Comm_free(&active_comm_);
  }
}

bool TestTaskALL::ValidationImpl() {
  if (rank_ != 0) {
    return true;  // Non-root processes assume valid
  }

  if (task_data->inputs.empty() || task_data->inputs_count.empty() || task_data->outputs.empty() ||
      task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->inputs.size() != 1 || task_data->inputs_count.size() != 1 || task_data->outputs.size() != 2 ||
      task_data->outputs_count.size() != 2) {
    return false;
  }
  if (task_data->inputs_count[0] == 0) return false;      // No points
  if (task_data->inputs_count[0] % 2 != 0) return false;  // Odd number of coordinates

  size_t num_points = task_data->inputs_count[0] / 2;
  if (num_points < 3) {
    // Original alputov handles <3 points by producing empty/small hull.
    // For this MPI version, let's enforce >=3 for a meaningful distributed computation.
    // If problem allows <3 points, RunImpl should handle it (e.g. rank 0 does all work).
    // For now, align with common expectation for convex hull algorithms.
    return false;
  }

  if (task_data->outputs_count[0] != 1) return false;  // hull_size output
  // output_count[1] is num_doubles for hull points, check against max possible (num_points * 2)
  if (task_data->outputs_count[1] < num_points * 2) return false;

  return true;
}

bool TestTaskALL::PreProcessingImpl() {
  if (rank_ == 0) {
    size_t num_input_doubles = task_data->inputs_count[0];
    size_t num_points = num_input_doubles / 2;
    input_points_.resize(num_points);
    auto* input_doubles = reinterpret_cast<double*>(task_data->inputs[0]);
    for (size_t i = 0; i < num_points; ++i) {
      input_points_[i] = Point(input_doubles[2 * i], input_doubles[2 * i + 1]);
    }
  }
  // No MPI communication needed here yet, data stays on rank 0.
  return true;
}

bool TestTaskALL::RunImpl() {
  size_t total_num_points = 0;
  if (rank_ == 0) {
    total_num_points = input_points_.size();
  }
  MPI_Bcast(&total_num_points, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (total_num_points < 3) {
    if (rank_ == 0) {
      // Handle cases with < 3 points (e.g., all points are the hull or empty hull)
      // This simplified logic assumes if < 3 points, they are the hull (or empty if 0)
      // Original alputov logic for small N:
      // 0 points: empty hull
      // 1 point: that point is the hull
      // 2 points: those two points are the hull (line segment)
      if (total_num_points == 0) {
        convex_hull_.clear();
      } else if (total_num_points == 1) {
        convex_hull_ = {input_points_[0]};
      } else if (total_num_points == 2) {
        // Ensure a consistent order if needed, e.g., by y then x.
        // Or just take them as is. For 2 points, order for hull might not matter.
        // Graham scan specific order might start with min-y.
        // Let's sort them to have a canonical 2-point hull.
        if (input_points_[0] < input_points_[1]) {
          convex_hull_ = {input_points_[0], input_points_[1]};
        } else {
          convex_hull_ = {input_points_[1], input_points_[0]};
        }
      }
    }
    return true;  // All processes return, rank 0 has the result.
  }

  // Determine active processes and create communicator
  active_procs_count_ = std::min(world_size_, static_cast<int>(total_num_points));
  if (active_procs_count_ == 0 && total_num_points > 0) active_procs_count_ = 1;  // at least one if points exist

  int color = (rank_ < active_procs_count_) ? 0 : 1;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank_, &active_comm_);

  if (color == 1) {  // This process is not active
    return true;
  }
  // Update rank_ and world_size_ for the active_comm_
  MPI_Comm_rank(active_comm_, &rank_);  // rank_ is now 0..active_procs_count-1
  // MPI_Comm_size(active_comm_, &active_procs_count_); // active_procs_count_ already set

  // 1. Find Pivot (on rank 0 of active_comm_) and Broadcast
  if (rank_ == 0) {
    pivot_ = FindPivot(input_points_);
  }
  MPI_Bcast(&pivot_, 1, mpi_point_datatype_, 0, active_comm_);

  // 2. Prepare and Scatter Points (excluding pivot)
  std::vector<Point> points_to_sort_globally;
  if (rank_ == 0) {
    for (const auto& p : input_points_) {
      if (p != pivot_) {
        points_to_sort_globally.push_back(p);
      }
    }
  }

  size_t num_points_to_scatter = 0;
  if (rank_ == 0) {
    num_points_to_scatter = points_to_sort_globally.size();
  }
  MPI_Bcast(&num_points_to_scatter, 1, MPI_UNSIGNED_LONG, 0, active_comm_);

  if (num_points_to_scatter == 0) {  // All points were the pivot or duplicates of it
    if (rank_ == 0) {
      convex_hull_ = {pivot_};  // Hull is just the pivot
    }
    return true;
  }

  std::vector<int> send_counts(active_procs_count_);
  std::vector<int> displs(active_procs_count_);
  int local_recv_count = 0;

  if (rank_ == 0) {
    int base_count = num_points_to_scatter / active_procs_count_;
    int remainder = num_points_to_scatter % active_procs_count_;
    displs[0] = 0;
    for (int i = 0; i < active_procs_count_; ++i) {
      send_counts[i] = base_count + (i < remainder ? 1 : 0);
      if (i > 0) {
        displs[i] = displs[i - 1] + send_counts[i - 1];
      }
    }
  }
  MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_recv_count, 1, MPI_INT, 0, active_comm_);

  local_points_.resize(local_recv_count);
  MPI_Scatterv(rank_ == 0 ? points_to_sort_globally.data() : nullptr, send_counts.data(), displs.data(),
               mpi_point_datatype_, local_points_.data(), local_recv_count, mpi_point_datatype_, 0, active_comm_);

  // 3. Local Sort (each active process sorts its local_points_)
  if (!local_points_.empty()) {
    LocalParallelSort(local_points_, pivot_);
  }

  // 4. Global Merge (Gather all sorted chunks to rank 0 and merge)
  std::vector<int> local_sizes(active_procs_count_);
  int my_local_size_after_sort = static_cast<int>(local_points_.size());  // Should be same as local_recv_count
  MPI_Gather(&my_local_size_after_sort, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, 0, active_comm_);

  int total_sorted_points_count = 0;
  if (rank_ == 0) {
    globally_sorted_points_.clear();
    displs[0] = 0;
    for (int i = 0; i < active_procs_count_; ++i) {
      total_sorted_points_count += local_sizes[i];
      if (i > 0) {
        displs[i] = displs[i - 1] + local_sizes[i - 1];
      }
    }
    globally_sorted_points_.resize(total_sorted_points_count);
  }

  MPI_Gatherv(local_points_.data(), my_local_size_after_sort, mpi_point_datatype_, globally_sorted_points_.data(),
              local_sizes.data(), displs.data(), mpi_point_datatype_, 0, active_comm_);

  // 5. Rank 0: Final Merge, Remove Duplicates, Build Hull
  if (rank_ == 0) {
    // The globally_sorted_points_ vector now contains concatenated sorted chunks.
    // A full sort or a k-way merge is needed if chunks weren't perfectly divided for parallel merge sort.
    // Since MPI_Gatherv just concatenates, and local sorts were independent, a final sort/merge is necessary.
    // For simplicity with CompareAngles requiring pivot:
    if (total_sorted_points_count > 0) {
      LocalParallelSort(globally_sorted_points_, pivot_);  // Sort the whole gathered collection
      RemoveDuplicates(globally_sorted_points_);
    }

    if (globally_sorted_points_.empty()) {  // All points were pivot or duplicates
      convex_hull_ = {pivot_};
    } else {
      convex_hull_ = BuildHull(globally_sorted_points_, pivot_);
    }
  }
  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  if (rank_ == 0) {  // Only rank 0 (of active_comm_, which is also rank 0 of MPI_COMM_WORLD if active)
    int* hull_size_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    *hull_size_ptr = static_cast<int>(convex_hull_.size());

    double* hull_data_ptr = reinterpret_cast<double*>(task_data->outputs[1]);
    for (size_t i = 0; i < convex_hull_.size(); ++i) {
      hull_data_ptr[2 * i] = convex_hull_[i].x;
      hull_data_ptr[2 * i + 1] = convex_hull_[i].y;
    }
  }
  return true;
}

// --- Static helper implementations (from alputov_i_graham_scan_stl) ---

Point TestTaskALL::FindPivot(const std::vector<Point>& points) {
  if (points.empty()) {
    throw std::runtime_error("Cannot find pivot in empty set of points.");
  }
  return *std::min_element(points.begin(), points.end());
}

double TestTaskALL::CrossProduct(const Point& o, const Point& a, const Point& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

bool TestTaskALL::CompareAngles(const Point& p1, const Point& p2, const Point& pivot) {
  double cross = CrossProduct(pivot, p1, p2);
  constexpr double kEpsilon = 1e-9;  // Tolerance for floating point comparisons

  if (std::abs(cross) < kEpsilon) {  // Collinear
    // If collinear, closer point comes first
    double dist1_sq = (p1.x - pivot.x) * (p1.x - pivot.x) + (p1.y - pivot.y) * (p1.y - pivot.y);
    double dist2_sq = (p2.x - pivot.x) * (p2.x - pivot.x) + (p2.y - pivot.y) * (p2.y - pivot.y);
    return dist1_sq < dist2_sq;
  }
  return cross > 0;  // p1 is counter-clockwise from p2
}

void TestTaskALL::RemoveDuplicates(std::vector<Point>& points) {
  if (points.empty()) return;
  // Sort to bring duplicates together if not already (though angular sort + distance should handle it mostly)
  // std::sort(points.begin(), points.end()); // Using Point::operator< if needed before unique
  auto last = std::unique(points.begin(), points.end());  // Uses Point::operator==
  points.erase(last, points.end());
}

std::vector<Point> TestTaskALL::BuildHull(const std::vector<Point>& sorted_points, const Point& pivot) {
  std::vector<Point> hull;
  if (sorted_points.empty()) {  // Only pivot was present
    hull.push_back(pivot);
    return hull;
  }

  hull.push_back(pivot);
  hull.push_back(sorted_points[0]);

  if (sorted_points.size() == 1) {  // Pivot + 1 other point
    return hull;
  }

  for (size_t i = 1; i < sorted_points.size(); ++i) {
    const Point& p = sorted_points[i];
    // Use a small epsilon for cross product check to handle floating point inaccuracies
    // A value < -epsilon means a right turn (or straight if very close to 0)
    while (hull.size() >= 2 &&
           CrossProduct(hull[hull.size() - 2], hull.back(), p) < 1e-9) {  // <=0 for strict, <eps for tolerance
      hull.pop_back();
    }
    hull.push_back(p);
  }

  // Final check for collinearity with the first point of the hull (pivot)
  // This handles cases where the last point added makes the segment with pivot collinear with the first actual segment
  // Not strictly needed if sorting and cross product are robust, but good for safety.
  while (hull.size() >= 3 && CrossProduct(hull[hull.size() - 2], hull.back(), hull[0]) < 1e-9) {
    hull.pop_back();
  }
  // Check for the very first segment too with the last point
  if (hull.size() >= 3 && CrossProduct(hull.back(), hull[0], hull[1]) < 1e-9) {
    hull.erase(hull.begin());  // remove hull[0] which is pivot, if hull[last], hull[0], hull[1] are collinear
                               // This logic needs care. Simpler is that the hull is {pivot, p1, p2 ...}
                               // The original alputov included pivot in the hull.
  }

  return hull;
}

void TestTaskALL::LocalParallelSort(std::vector<Point>& points, const Point& pivot_for_sort) {
  if (points.size() <= 1) return;

  auto comparator = [&](const Point& a, const Point& b) { return CompareAngles(a, b, pivot_for_sort); };

  const size_t n = points.size();
  // Use GetPPCNumThreads() or a fixed number of threads.
  // std::thread::hardware_concurrency() can be an option.
  // For ppc framework, use ppc::util::GetPPCNumThreads().
  const size_t num_threads = std::min(n, static_cast<size_t>(ppc::util::GetPPCNumThreads()));

  if (num_threads <= 1 || n < 1000) {  // Threshold for parallel sort overhead
    std::sort(points.begin(), points.end(), comparator);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<size_t> chunk_offsets(num_threads + 1);

  for (size_t i = 0; i <= num_threads; ++i) {
    chunk_offsets[i] = (i * n) / num_threads;
  }

  using DifferenceType = typename std::vector<Point>::difference_type;

  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&points, comparator, start_offset = chunk_offsets[i], end_offset = chunk_offsets[i + 1]]() {
      auto start_it = std::next(points.begin(), static_cast<DifferenceType>(start_offset));
      auto end_it = std::next(points.begin(), static_cast<DifferenceType>(end_offset));
      std::sort(start_it, end_it, comparator);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Merge sorted chunks
  // This performs (num_threads - 1) merges.
  // A more optimized k-way merge could be used for many threads, but this is standard.
  for (size_t i = 1; i < num_threads; ++i) {
    auto first_unsorted_chunk_start = std::next(points.begin(), static_cast<DifferenceType>(chunk_offsets[i]));
    auto first_unsorted_chunk_end = std::next(points.begin(), static_cast<DifferenceType>(chunk_offsets[i + 1]));
    // The merge up to chunk_offsets[i] is already sorted with points.begin()
    std::inplace_merge(points.begin(), first_unsorted_chunk_start, first_unsorted_chunk_end, comparator);
  }
}

}  // namespace alputov_i_graham_scan_all