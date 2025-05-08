#include "all/plekhanov_d_dijkstra/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
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
      if (i < graph_data.size() && graph_data[i] != -1) return false;
      break;
    }
    size_t dest = graph_data[i];
    int weight = graph_data[i + 1];
    if (weight < 0) {
      return false;
    }
    if (dest < num_vertices) {
      graph[current_vertex].emplace_back(static_cast<int>(dest), weight);
    } else {
      return false;
    }
    i += 2;
  }
  if (i < graph_data.size()) return false;
  if (current_vertex != num_vertices) return false;
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

  if (start_vertex_ < 0 || start_vertex_ >= static_cast<int>(num_vertices_)) {
    return false;
  }

  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::RunImpl() {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  
  std::vector<std::vector<std::pair<int, int>>> local_graph;
  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, local_graph)) {
    return false;
  }

  int vertices_per_proc = (num_vertices_ + size - 1) / size;

  std::vector<int> local_distances(num_vertices_, INT_MAX);
  if (rank == 0) {
    local_distances[start_vertex_] = 0;
  }

  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
  if (rank == 0) {
    pq.emplace(0, start_vertex_);
  }

  std::mutex pq_mutex;
  bool done = false;
  int iteration = 0;
  const int max_iterations = num_vertices_;

  while (!done && iteration < max_iterations) {
    iteration++;
    
    while (true) {
      int u = -1;
      int cur_dist = -1;

      {
        std::lock_guard<std::mutex> lock(pq_mutex);
        if (!pq.empty()) {
          auto top = pq.top();
          pq.pop();
          u = top.second;
          cur_dist = top.first;
        } else {
          break;
        }
      }

      if (u != -1 && cur_dist == local_distances[u]) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < local_graph[u].size(); i++) {
          const auto& edge = local_graph[u][i];
          int v = edge.first;
          int weight = edge.second;

          if (local_distances[u] != INT_MAX) {
            int new_dist = local_distances[u] + weight;
            if (new_dist < local_distances[v]) {
              local_distances[v] = new_dist;
              std::lock_guard<std::mutex> lock(pq_mutex);
              pq.emplace(new_dist, v);
            }
          }
        }
      }
    }

    std::vector<std::vector<int>> all_distances;
    boost::mpi::all_gather(world, local_distances, all_distances);

    bool updated = false;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < vertices_per_proc; j++) {
        int global_vertex = i * vertices_per_proc + j;
        if (global_vertex < num_vertices_) {
          int old_dist = local_distances[global_vertex];
          local_distances[global_vertex] = std::min(local_distances[global_vertex], 
                                                  all_distances[i][global_vertex]);
          if (old_dist != local_distances[global_vertex]) {
            updated = true;
            std::lock_guard<std::mutex> lock(pq_mutex);
            pq.emplace(local_distances[global_vertex], global_vertex);
          }
        }
      }
    }

    int local_empty = pq.empty() ? 1 : 0;
    int global_empty;
    boost::mpi::all_reduce(world, local_empty, global_empty, std::plus<int>());
    done = (global_empty == size) && !updated;
  }

  std::vector<std::vector<int>> gathered;
  boost::mpi::gather(world, local_distances, gathered, 0);

  if (rank == 0) {
    distances_.assign(num_vertices_, INT_MAX);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < vertices_per_proc; ++j) {
        int global_vertex = i * vertices_per_proc + j;
        if (global_vertex < num_vertices_) {
          distances_[global_vertex] = std::min(distances_[global_vertex], gathered[i][global_vertex]);
        }
      }
    }
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