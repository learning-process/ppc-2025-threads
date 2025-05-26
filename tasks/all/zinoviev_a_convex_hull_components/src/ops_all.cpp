#include "all/zinoviev_a_convex_hull_components/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using namespace zinoviev_a_convex_hull_components_all;

ConvexHullMPI::ConvexHullMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), rank_(-1), comm_size_(-1) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
}

bool ConvexHullMPI::PreProcessingImpl() noexcept {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  const auto* global_input_data = reinterpret_cast<const int*>(task_data->inputs[0]);
  const int global_width = static_cast<int>(task_data->inputs_count[0]);
  const int global_height = static_cast<int>(task_data->inputs_count[1]);

  local_components_.clear();
  auto local_ranges = CalculateLocalRanges(global_height);

  int local_start_row = local_ranges[rank_ * 2];
  int local_end_row = local_ranges[(rank_ * 2) + 1];
  int local_height = local_end_row - local_start_row;
  int local_width = global_width;

  if (local_height <= 0) {
    return true;
  }

  std::vector<int> local_input_data(static_cast<size_t>(local_width) * static_cast<size_t>(local_height));

#pragma omp parallel for
  for (int y = 0; y < local_height; ++y) {
    for (int x = 0; x < local_width; ++x) {
      local_input_data[(static_cast<size_t>(y) * static_cast<size_t>(local_width)) + static_cast<size_t>(x)] =
          global_input_data[((local_start_row + y) * global_width) + x];
    }
  }

  std::vector<bool> visited(static_cast<size_t>(local_width) * static_cast<size_t>(local_height), false);
  std::vector<std::vector<Point>> thread_components;

#pragma omp parallel
  {
    std::vector<std::vector<Point>> local_thread_components;

#pragma omp for schedule(dynamic)
    for (int y = 0; y < local_height; ++y) {
      for (int x = 0; x < local_width; ++x) {
        size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(local_width)) + static_cast<size_t>(x);

        bool should_process = false;
#pragma omp critical
        {
          if (!visited[idx] && local_input_data[idx] != 0) {
            visited[idx] = true;
            should_process = true;
          }
        }

        if (should_process) {
          std::vector<Point> component;
          BFS(local_input_data.data(), local_width, local_height, x, y, visited, component);

          for (auto& p : component) {
            p.y = p.y + local_start_row;
          }
          local_thread_components.push_back(component);
        }
      }
    }

#pragma omp critical
    {
      thread_components.insert(thread_components.end(), local_thread_components.begin(), local_thread_components.end());
    }
  }

  local_components_ = std::move(thread_components);
  return true;
}

std::vector<int> ConvexHullMPI::CalculateLocalRanges(int global_size) const noexcept {
  std::vector<int> ranges(static_cast<size_t>(comm_size_) * 2);
  int base_size = global_size / comm_size_;
  int remainder = global_size % comm_size_;
  int current_start = 0;

  for (int i = 0; i < comm_size_; ++i) {
    int size = base_size + (i < remainder ? 1 : 0);
    ranges[static_cast<size_t>(i) * 2] = current_start;
    ranges[(static_cast<size_t>(i) * 2) + 1] = current_start + size;
    current_start += size;
  }

  return ranges;
}

void ConvexHullMPI::BFS(const int* local_input_data, int local_width, int local_height, int start_x, int start_y,
                        std::vector<bool>& visited, std::vector<Point>& component) noexcept {
  std::queue<Point> queue;
  queue.push({start_x, start_y});

  constexpr int kDx[] = {-1, 1, 0, 0};
  constexpr int kDy[] = {0, 0, -1, 1};

  while (!queue.empty()) {
    Point p = queue.front();
    queue.pop();
    component.push_back(p);

    for (int i = 0; i < 4; ++i) {
      int nx = p.x + kDx[i];
      int ny = p.y + kDy[i];

      if (nx >= 0 && nx < local_width && ny >= 0 && ny < local_height) {
        size_t nidx = (static_cast<size_t>(ny) * static_cast<size_t>(local_width)) + static_cast<size_t>(nx);

        bool should_add = false;
#pragma omp critical
        {
          if (!visited[nidx] && local_input_data[nidx] != 0) {
            visited[nidx] = true;
            should_add = true;
          }
        }

        if (should_add) {
          queue.push({nx, ny});
        }
      }
    }
  }
}

bool ConvexHullMPI::ValidationImpl() noexcept {
  return task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 1 &&
         task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

int ConvexHullMPI::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

std::vector<Point> ConvexHullMPI::FindConvexHull(const std::vector<Point>& points) noexcept {
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> sorted_points(points);

#pragma omp parallel
  {
#pragma omp single
    {
      std::ranges::sort(sorted_points,
                        [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    }
  }

  std::vector<Point> hull;
  hull.reserve(sorted_points.size() * 2);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  int lower_size = static_cast<int>(hull.size()) + 1;
  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (static_cast<int>(hull.size()) >= lower_size && Cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (hull.size() > 1) {
    hull.pop_back();
  }

  return hull;
}

bool ConvexHullMPI::RunImpl() noexcept {
  std::vector<Point> all_points;

#pragma omp parallel
  {
    std::vector<Point> thread_points;

#pragma omp for
    for (int i = 0; i < static_cast<int>(local_components_.size()); ++i) {
      const auto& component = local_components_[static_cast<size_t>(i)];
      thread_points.insert(thread_points.end(), component.begin(), component.end());
    }

#pragma omp critical
    { all_points.insert(all_points.end(), thread_points.begin(), thread_points.end()); }
  }

  int local_count = static_cast<int>(all_points.size());
  std::vector<int> counts(comm_size_);

  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(comm_size_);
  int total_count = 0;
  for (int i = 0; i < comm_size_; ++i) {
    displs[i] = total_count;
    total_count += counts[i];
  }

  std::vector<Point> global_points(total_count);

  std::vector<int> byte_counts(comm_size_);
  std::vector<int> byte_displs(comm_size_);
  for (int i = 0; i < comm_size_; ++i) {
    byte_counts[i] = counts[i] * static_cast<int>(sizeof(Point));
    byte_displs[i] = displs[i] * static_cast<int>(sizeof(Point));
  }

  MPI_Allgatherv(all_points.data(), local_count * static_cast<int>(sizeof(Point)), MPI_BYTE, global_points.data(),
                 byte_counts.data(), byte_displs.data(), MPI_BYTE, MPI_COMM_WORLD);

  if (!global_points.empty()) {
    final_hull_ = FindConvexHull(global_points);
  }

  return true;
}

bool ConvexHullMPI::PostProcessingImpl() noexcept {
  if (task_data->outputs.empty()) {
    return false;
  }

  if (rank_ == 0) {
    if (task_data->outputs[0] == nullptr || task_data->outputs_count.empty()) {
      return false;
    }

    auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
    size_t allocated_output_size = task_data->outputs_count[0];
    size_t points_to_copy = std::min(final_hull_.size(), allocated_output_size);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(points_to_copy); ++i) {
      output_ptr[static_cast<size_t>(i)] = final_hull_[static_cast<size_t>(i)];
    }

    task_data->outputs_count[0] = points_to_copy;
  } else {
    if (!task_data->outputs.empty() && !task_data->outputs_count.empty()) {
      task_data->outputs_count[0] = 0;
    }
  }

  return true;
}