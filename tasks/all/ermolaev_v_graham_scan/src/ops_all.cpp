#include "all/ermolaev_v_graham_scan/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

int ermolaev_v_graham_scan_all::TestTaskALL::CrossProduct(const Point &p1, const Point &p2, const Point &p3) {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
}

size_t ermolaev_v_graham_scan_all::TestTaskALL::IndexOfMinElement() {
  size_t min_idx = 0;
  int input_size = static_cast<int>(input_.size());

#pragma omp parallel
  {
    size_t local_min_idx = 0;

#pragma omp for nowait
    for (int i = 1; i < input_size; i++) {
      if (input_[i] < input_[local_min_idx]) {
        local_min_idx = i;
      }
    }

#pragma omp critical
    {
      if (input_[local_min_idx] < input_[min_idx]) {
        min_idx = local_min_idx;
      }
    }
  }

  return min_idx;
}

bool ermolaev_v_graham_scan_all::TestTaskALL::IsAllCollinear() {
  int input_size = static_cast<int>(input_.size());
  bool found_non_collinear = false;

#pragma omp parallel for reduction(|| : found_non_collinear)
  for (int i = 0; i < input_size - 2; ++i) {
    if (!found_non_collinear) {
      for (int j = i + 1; j < input_size - 1; ++j) {
        if (!found_non_collinear) {
          for (int k = j + 1; k < input_size; ++k) {
            if (CrossProduct(input_[i], input_[j], input_[k]) != 0) {
              found_non_collinear = true;
            }
          }
        }
      }
    }
  }

  return !found_non_collinear;
}

bool ermolaev_v_graham_scan_all::TestTaskALL::IsAllSame() {
  const Point &first = input_[0];
  bool all_same = true;
  int input_size = static_cast<int>(input_.size());

#pragma omp parallel for reduction(&& : all_same)
  for (int i = 1; i < input_size; ++i) {
    if (input_[i] != first) {
      all_same = false;
    }
  }

  return all_same;
}

bool ermolaev_v_graham_scan_all::TestTaskALL::CheckGrahamNecessaryConditions() {
  if (input_.size() < kMinInputPoints) {
    return false;
  }

  return !IsAllSame() && !IsAllCollinear();
}

void ermolaev_v_graham_scan_all::TestTaskALL::GrahamScan() {
  output_.clear();
  output_.emplace_back(input_[0]);
  output_.emplace_back(input_[1]);

  Point p1;
  Point p2;
  Point p3;
  for (size_t i = kMinStackPoints; i < input_.size(); i++) {
    while (output_.size() >= kMinStackPoints) {
      p1 = output_[output_.size() - 2];
      p2 = output_[output_.size() - 1];
      p3 = input_[i];

      int cross = CrossProduct(p1, p2, p3);

      if (cross > 0) {
        break;
      }
      output_.pop_back();
    }
    output_.emplace_back(input_[i]);
  }
}

bool ermolaev_v_graham_scan_all::TestTaskALL::PreProcessingImpl() {
  int rank = world_.rank();

  if (rank == 0) {
    auto *in_ptr = reinterpret_cast<Point *>(task_data->inputs[0]);
    input_ = std::vector<Point>(in_ptr, in_ptr + task_data->inputs_count[0]);
  }

  output_ = std::vector<Point>();
  return true;
}

bool ermolaev_v_graham_scan_all::TestTaskALL::ValidationImpl() {
  return world_.rank() != 0 ||
         (task_data->inputs_count[0] >= kMinInputPoints && task_data->inputs_count[0] <= task_data->outputs_count[0]);
}

bool ermolaev_v_graham_scan_all::TestTaskALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  Point min_point;
  if (rank == 0) {
    if (!CheckGrahamNecessaryConditions()) {
      return false;
    }

    size_t min_idx = IndexOfMinElement();
    std::iter_swap(input_.begin(), input_.begin() + static_cast<int>(min_idx));
    min_point = input_[0];

    int points_per_proc = static_cast<int>(input_.size()) / size;
    int remainder = static_cast<int>(input_.size()) % size;

    int start_idx = 1;
    int end_idx = points_per_proc + (remainder > 0 ? 1 : 0);
    local_points_.assign(input_.begin() + start_idx, input_.begin() + end_idx);

    int current_idx = end_idx;
    for (int i = 1; i < size; i++) {
      int current_points = points_per_proc + (i < remainder ? 1 : 0);
      std::vector<Point> to_send(input_.begin() + current_idx, input_.begin() + current_idx + current_points);
      world_.send(i, 0, to_send);
      current_idx += current_points;
    }
  } else {
    world_.recv(0, 0, local_points_);
  }
  boost::mpi::broadcast(world_, min_point, 0);

  auto comp = [&](const Point &a, const Point &b) {
    auto squared_dist = [](const Point &p1, const Point &p2) -> int {
      int dx = p1.x - p2.x;
      int dy = p1.y - p2.y;
      return ((dx * dx) + (dy * dy));
    };

    int cross = CrossProduct(min_point, a, b);
    if (cross == 0) {
      return squared_dist(a, min_point) < squared_dist(b, min_point);
    }

    return cross > 0;
  };

  ParallelSort(local_points_.begin(), local_points_.end(), comp);

  if (rank == 0) {
    std::vector<Point> sorted_points(local_points_.size());
    std::vector<Point> buffer;

    for (int i = 1; i < size; i++) {
      world_.recv(i, 0, sorted_points);
      std::ranges::merge(sorted_points, local_points_, std::back_inserter(buffer), comp);
      local_points_.swap(buffer);
      buffer.clear();
    }

    input_.assign(local_points_.begin(), local_points_.end());
    input_.insert(input_.begin(), min_point);

    GrahamScan();
  } else {
    world_.send(0, 0, local_points_);
  }

  return true;
}

bool ermolaev_v_graham_scan_all::TestTaskALL::PostProcessingImpl() {
  int rank = world_.rank();
  if (rank != 0) {
    return true;
  }

  task_data->outputs_count.clear();
  task_data->outputs_count.push_back(output_.size());
  std::ranges::copy(output_, reinterpret_cast<Point *>(task_data->outputs[0]));
  return true;
}