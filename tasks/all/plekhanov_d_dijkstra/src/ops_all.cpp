#include "all/plekhanov_d_dijkstra/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace boost {
namespace mpi {
template <>
struct is_mpi_datatype<std::pair<int, int>> : public true_type {};
}  // namespace mpi
}  // namespace boost

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
  std::vector<std::vector<std::pair<int, int>>> graph;
  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, graph)) {
    return false;
  }

  std::vector<int> local_distances(num_vertices_, INT_MAX);
  local_distances[start_vertex_] = 0;

  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
  pq.emplace(0, start_vertex_);
  std::mutex pq_mutex;

#pragma omp parallel shared(graph, local_distances, pq, pq_mutex) default(none)
  {
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

      if (u != -1) {
        for (const auto& edge : graph[u]) {
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
  }

  std::vector<std::vector<int>> gathered;
  boost::mpi::gather(world, local_distances, gathered, 0);

  if (world.rank() == 0) {
    distances_.assign(num_vertices_, INT_MAX);
    for (int i = 0; i < static_cast<int>(num_vertices_); ++i) {
      for (const auto& proc_dists : gathered) {
        if (static_cast<size_t>(i) < proc_dists.size()) {
          distances_[i] = std::min(distances_[i], proc_dists[i]);
        }
      }
    }
  }

  boost::mpi::broadcast(world, distances_, 0);
  return true;
}

bool plekhanov_d_dijkstra_all::TestTaskALL::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}