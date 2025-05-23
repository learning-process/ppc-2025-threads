#include "all/zinoviev_a_convex_hull_components/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
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

  const auto* global_input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int global_width = static_cast<int>(task_data->inputs_count[0]);
  const int global_height = static_cast<int>(task_data->inputs_count[1]);

  local_components_.clear();

  auto local_ranges = CalculateLocalRanges(global_height);
  int local_start_row = local_ranges[rank_ * 2];
  int local_end_row = local_ranges[(rank_ * 2) + 1];
  int local_height = local_end_row - local_start_row;
  int local_width = global_width;
  std::vector<int> local_input_data(static_cast<size_t>(local_width) * static_cast<size_t>(local_height));

  for (int y = 0; y < local_height; ++y) {
    for (int x = 0; x < local_width; ++x) {
      local_input_data[(static_cast<size_t>(y) * static_cast<size_t>(local_width)) + static_cast<size_t>(x)] =
          global_input_data[((local_start_row + y) * global_width) + x];
    }
  }

  std::vector<bool> visited(static_cast<size_t>(local_width) * static_cast<size_t>(local_height), false);

  for (int y = 0; y < local_height; ++y) {
    for (int x = 0; x < local_width; ++x) {
      size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(local_width)) + static_cast<size_t>(x);
      if (!visited[idx] && local_input_data[idx] != 0) {
        std::vector<Point> component;
        BFS(local_input_data.data(), local_width, local_height, x, y, visited, component);
        for (auto& p : component) {
          p.y = p.y + local_start_row;
        }
        local_components_.push_back(component);
      }
    }
  }

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
  size_t start_idx = (static_cast<size_t>(start_y) * static_cast<size_t>(local_width)) + static_cast<size_t>(start_x);
  visited[start_idx] = true;

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
        if (!visited[nidx] && local_input_data[nidx] != 0) {
          visited[nidx] = true;
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
  std::ranges::sort(sorted_points,
                    [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

  std::vector<Point> hull;
  hull.reserve(sorted_points.size() * 2);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (!hull.empty()) {
    hull.pop_back();
  }

  return hull;
}

bool ConvexHullMPI::RunImpl() noexcept {
  local_hulls_.resize(local_components_.size());
  for (size_t i = 0; i < local_components_.size(); ++i) {
    local_hulls_[i] = FindConvexHull(local_components_[i]);
  }

  int num_local_components = static_cast<int>(local_components_.size());
  std::vector<int> send_counts(static_cast<size_t>(comm_size_));
  MPI_Allgather(&num_local_components, 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displacements(static_cast<size_t>(comm_size_), 0);
  int total_components = 0;
  for (int i = 0; i < comm_size_; ++i) {
    displacements[static_cast<size_t>(i)] = total_components;
    total_components += send_counts[static_cast<size_t>(i)];
  }

  global_components_.resize(static_cast<size_t>(total_components));
  std::vector<int> recv_counts(static_cast<size_t>(comm_size_));
  std::vector<MPI_Request> send_requests(local_components_.size());
  std::vector<MPI_Request> recv_requests(static_cast<size_t>(total_components));
  std::vector<MPI_Status> statuses(static_cast<size_t>(total_components) + local_components_.size());

  std::vector<std::vector<Point>> all_local_components(static_cast<size_t>(total_components));
  std::vector<int> component_sizes(local_components_.size());
  std::vector<int> all_component_sizes(static_cast<size_t>(total_components));
  std::vector<int> component_displacements(static_cast<size_t>(comm_size_), 0);
  std::vector<int> all_component_displacements(static_cast<size_t>(total_components), 0);
  int current_displacement = 0;
  for (int i = 0; i < comm_size_; ++i) {
    component_displacements[static_cast<size_t>(i)] = current_displacement;
    current_displacement += send_counts[static_cast<size_t>(i)];
  }

  for (size_t i = 0; i < local_components_.size(); ++i) {
    component_sizes[i] = static_cast<int>(local_components_[i].size());
  }
  MPI_Allgatherv(component_sizes.data(), static_cast<int>(local_components_.size()), MPI_INT,
                 all_component_sizes.data(), send_counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

  std::vector<Point> send_buffer;
  std::vector<int> send_buffer_sizes(local_components_.size());
  std::vector<int> send_buffer_displacements(local_components_.size(), 0);
  int current_send_offset = 0;
  for (size_t i = 0; i < local_components_.size(); ++i) {
    send_buffer_sizes[i] = static_cast<int>(local_components_[i].size() * sizeof(Point));
    send_buffer_displacements[i] = current_send_offset;
    current_send_offset += send_buffer_sizes[i];
    send_buffer.insert(send_buffer.end(), local_components_[i].begin(), local_components_[i].end());
  }

  std::vector<Point> recv_buffer(
      static_cast<size_t>(std::accumulate(all_component_sizes.begin(), all_component_sizes.end(), 0)));
  std::vector<int> recv_buffer_sizes(static_cast<size_t>(total_components));
  std::vector<int> recv_buffer_displacements(static_cast<size_t>(total_components), 0);
  int current_recv_offset = 0;
  int global_component_index = 0;
  for (int i = 0; i < comm_size_; ++i) {
    for (int j = 0; j < send_counts[static_cast<size_t>(i)]; ++j) {
      size_t idx = static_cast<size_t>(displacements[i]) + static_cast<size_t>(j);
      recv_buffer_sizes[static_cast<size_t>(global_component_index)] =
          static_cast<int>(all_component_sizes[idx] * sizeof(Point));
      recv_buffer_displacements[static_cast<size_t>(global_component_index)] = current_recv_offset;
      current_recv_offset += recv_buffer_sizes[static_cast<size_t>(global_component_index)];
      global_component_index++;
    }
  }

  MPI_Allgatherv(send_buffer.data(), static_cast<int>(send_buffer.size() * sizeof(Point)), MPI_BYTE, recv_buffer.data(),
                 recv_buffer_sizes.data(), recv_buffer_displacements.data(), MPI_BYTE, MPI_COMM_WORLD);

  int recv_buffer_index = 0;
  global_component_index = 0;
  for (int i = 0; i < comm_size_; ++i) {
    for (int j = 0; j < send_counts[static_cast<size_t>(i)]; ++j) {
      size_t idx = static_cast<size_t>(displacements[i]) + static_cast<size_t>(j);
      int num_points = all_component_sizes[idx];
      global_components_[static_cast<size_t>(global_component_index)].resize(static_cast<size_t>(num_points));
      std::copy(recv_buffer.begin() + recv_buffer_index, recv_buffer.begin() + recv_buffer_index + num_points,
                global_components_[static_cast<size_t>(global_component_index)].begin());
      recv_buffer_index += num_points;
      global_component_index++;
    }
  }

  global_hulls_.resize(global_components_.size());
  for (size_t i = 0; i < global_components_.size(); ++i) {
    global_hulls_[i] = FindConvexHull(global_components_[i]);
  }

  return true;
}

bool ConvexHullMPI::PostProcessingImpl() noexcept {
  if (rank_ != 0) {
    return true;
  }

  if (task_data->outputs.empty()) {
    return false;
  }

  size_t total_points = 0;
  std::vector<int> hull_sizes(global_hulls_.size());
  for (size_t i = 0; i < global_hulls_.size(); ++i) {
    hull_sizes[i] = static_cast<int>(global_hulls_[i].size());
    total_points += global_hulls_[i].size();
  }

  std::vector<int> all_hull_sizes(total_points);
  std::vector<int> all_hull_counts(static_cast<size_t>(comm_size_));
  std::vector<int> all_hull_displacements(static_cast<size_t>(comm_size_), 0);
  int num_global_hulls = static_cast<int>(global_hulls_.size());
  MPI_Gather(&num_global_hulls, 1, MPI_INT, all_hull_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    int current_displacement = 0;
    for (int i = 0; i < comm_size_; ++i) {
      all_hull_displacements[static_cast<size_t>(i)] = current_displacement;
      current_displacement += all_hull_counts[static_cast<size_t>(i)];
    }
  }

  std::vector<int> gathered_hull_sizes(total_points);
  std::vector<int> send_hull_sizes(global_hulls_.size());
  for (size_t i = 0; i < global_hulls_.size(); ++i) {
    send_hull_sizes[i] = static_cast<int>(global_hulls_[i].size());
  }
  std::vector<int> recv_counts_hulls(all_hull_counts.empty() ? 0 : *std::ranges::max_element(all_hull_counts), 0);
  std::vector<int> recv_displs_hulls(all_hull_counts.size(), 0);
  if (rank_ == 0) {
    int current_size = 0;
    for (size_t i = 0; i < all_hull_counts.size(); ++i) {
      recv_counts_hulls[i] = all_hull_counts[i];
      recv_displs_hulls[i] = current_size;
      current_size += all_hull_counts[i];
    }
  }

  std::vector<int> all_hulls_flat_sizes(total_points);
  MPI_Gatherv(send_hull_sizes.data(), static_cast<int>(send_hull_sizes.size()), MPI_INT, all_hulls_flat_sizes.data(),
              recv_counts_hulls.data(), recv_displs_hulls.data(), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<Point> all_hulls_flat;
  std::vector<int> send_hull_point_counts(global_hulls_.size());
  std::vector<int> send_hull_point_displs(global_hulls_.size(), 0);
  int current_point_offset = 0;
  std::vector<Point> local_hull_points_flat;
  for (const auto& hull : global_hulls_) {
    send_hull_point_counts.push_back(static_cast<int>(hull.size()));
    send_hull_point_displs.push_back(current_point_offset);
    local_hull_points_flat.insert(local_hull_points_flat.end(), hull.begin(), hull.end());
    current_point_offset += static_cast<int>(hull.size());
  }

  std::vector<int> recv_hull_point_counts(total_points, 0);
  std::vector<int> recv_hull_point_displs(all_hull_counts.size() + 1, 0);
  std::vector<Point> gathered_hull_points(total_points);
  if (rank_ == 0) {
    size_t hull_index = 0;
    for (size_t i = 0; i < all_hull_counts.size(); ++i) {
      for (int j = 0; j < all_hull_counts[i]; ++j) {
        recv_hull_point_counts[hull_index] = all_hulls_flat_sizes[recv_displs_hulls[i] + j];
        recv_hull_point_displs[i + 1] += recv_hull_point_counts[hull_index];
        hull_index++;
      }
    }
    for (size_t i = 1; i < recv_hull_point_displs.size(); ++i) {
      recv_hull_point_displs[i] += recv_hull_point_displs[i - 1];
    }
    gathered_hull_points.resize(static_cast<size_t>(recv_hull_point_displs.back()));
  }

  MPI_Gatherv(local_hull_points_flat.data(), static_cast<int>(local_hull_points_flat.size() * sizeof(Point)), MPI_BYTE,
              gathered_hull_points.data(), recv_hull_point_counts.data(), recv_hull_point_displs.data(), MPI_BYTE, 0,
              MPI_COMM_WORLD);

  if (rank_ == 0) {
    auto* output = reinterpret_cast<Point*>(task_data->outputs[0]);
    size_t offset = 0;
    for (const auto& point : gathered_hull_points) {
      output[offset++] = point;
    }
    task_data->outputs_count[0] = static_cast<int>(gathered_hull_points.size());
  } else if (!task_data->outputs.empty()) {
    task_data->outputs_count[0] = 0;
  }

  return true;
}