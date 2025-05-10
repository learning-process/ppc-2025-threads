#include "all/plekhanov_d_dijkstra/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace plekhanov_d_dijkstra_all {

namespace {

bool ConvertGraphToAdjacencyList(const std::vector<int>& graph_data, size_t num_vertices,
                                 std::vector<std::vector<std::pair<int, int>>>& graph) {
  graph.assign(num_vertices, {});
  size_t current_vertex = 0;
  size_t i = 0;
  while (i < graph_data.size() && current_vertex < num_vertices) {
    if (graph_data[i] == -1) {
      current_vertex++;
      i++;
      continue;
    }
    if (i + 1 >= graph_data.size()) {
      break;
    }
    size_t dest = graph_data[i];
    int weight = graph_data[i + 1];
    if (weight < 0) {
      return false;
    }
    if (dest < num_vertices) {
      graph[current_vertex].emplace_back(static_cast<int>(dest), weight);
    }
    i += 2;
  }
  return true;
}

}  // namespace

}  // namespace plekhanov_d_dijkstra_all

const int plekhanov_d_dijkstra_all::TestTaskALL::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_all::TestTaskALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::RunImpl() {
  std::vector<std::vector<std::pair<int, int>>> adj_list(num_vertices_);
  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, adj_list)) {
    return false;
  }

  const int INF = INT_MAX;
  std::vector<int> local_dist(num_vertices_, INF);
  local_dist[start_vertex_] = 0;

  const int chunk_size = (num_vertices_ + world_.size() - 1) / world_.size();
  const int start = world_.rank() * chunk_size;
  const int end = std::min(start + chunk_size, static_cast<int>(num_vertices_));

  bool updated = true;
  int iteration = 0;
  const int max_iterations = num_vertices_;

  while (updated && iteration < max_iterations) {
    updated = false;
    iteration++;

    #pragma omp parallel for schedule(dynamic)
    for (int u = start; u < end; ++u) {
      if (local_dist[u] == INF) continue;

      for (const auto& [neighbor, weight] : adj_list[u]) {
        int new_dist = local_dist[u] + weight;
        if (new_dist < local_dist[neighbor]) {
          #pragma omp critical
          {
            if (new_dist < local_dist[neighbor]) {
              local_dist[neighbor] = new_dist;
              updated = true;
            }
          }
        }
      }
    }

    std::vector<int> global_dist(num_vertices_);
    boost::mpi::all_reduce(world_, local_dist.data(), num_vertices_, global_dist.data(), boost::mpi::minimum<int>());

    #pragma omp parallel for
    for (int i = 0; i < num_vertices_; ++i) {
      if (global_dist[i] < local_dist[i]) {
        local_dist[i] = global_dist[i];
        updated = true;
      }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < num_vertices_; ++u) {
      if (local_dist[u] == INF) continue;

      for (const auto& [neighbor, weight] : adj_list[u]) {
        int new_dist = local_dist[u] + weight;
        if (new_dist < local_dist[neighbor]) {
          #pragma omp critical
          {
            if (new_dist < local_dist[neighbor]) {
              local_dist[neighbor] = new_dist;
              updated = true;
            }
          }
        }
      }
    }

    int local_updated = updated ? 1 : 0;
    int global_updated;
    boost::mpi::all_reduce(world_, local_updated, global_updated, std::plus<int>());
    updated = (global_updated > 0);
  }

  std::vector<int> final_dist(num_vertices_);
  boost::mpi::all_reduce(world_, local_dist.data(), num_vertices_, final_dist.data(), boost::mpi::minimum<int>());

  #pragma omp parallel for
  for (int i = 0; i < static_cast<int>(num_vertices_); ++i) {
    distances_[i] = final_dist[i];
  }

  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}