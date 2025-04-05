#include "omp/plekhanov_d_dijkstra/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

namespace plekhanov_d_dijkstra_omp {

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
    if (dest < num_vertices) graph[current_vertex].emplace_back(static_cast<int>(dest), weight);
    i += 2;
  }
  return true;
}

static int FindMinDistanceVertex(const std::vector<int>& distances, const std::vector<bool>& visited,
                                 size_t num_vertices) {
  int globalMin = INT_MAX, selectedVertex = -1;
#pragma omp parallel
  {
    int localMin = INT_MAX, localVertex = -1;
#pragma omp for nowait
    for (int v = 0; v < static_cast<int>(num_vertices); ++v) {
      if (!visited[v] && distances[v] < localMin) {
        localMin = distances[v];
        localVertex = v;
      }
    }
#pragma omp critical
    {
      if (localMin < globalMin) {
        globalMin = localMin;
        selectedVertex = localVertex;
      }
    }
  }
  return selectedVertex;
}

static void UpdateDistancesForVertex(int u, const std::vector<std::vector<std::pair<int, int>>>& graph,
                                     std::vector<int>& distances, const std::vector<bool>& visited) {
#pragma omp parallel for
  for (int j = 0; j < static_cast<int>(graph[u].size()); ++j) {
    int v = graph[u][j].first;
    int weight = graph[u][j].second;
    if (!visited[v] && distances[u] != INT_MAX) {
      int newDistance = distances[u] + weight;
#pragma omp critical
      {
        distances[v] = std::min(newDistance, distances[v]);
      }
    }
  }
}

}  // namespace plekhanov_d_dijkstra_omp

const int plekhanov_d_dijkstra_omp::TestTaskOpenMP::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr)
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  else
    start_vertex_ = 0;
  if (start_vertex_ < num_vertices_) distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::RunImpl() {
  std::vector<std::vector<std::pair<int, int>>> graph;
  if (!ConvertGraphToAdjacencyList(graph_data_, num_vertices_, graph)) {
    return false;
  }
  std::vector<bool> visited(num_vertices_, false);
  for (int count = 0; count < static_cast<int>(num_vertices_) - 1; ++count) {
    int u = FindMinDistanceVertex(distances_, visited, num_vertices_);
    if (u == -1) {
      break;
    }
    visited[u] = true;
    UpdateDistancesForVertex(u, graph, distances_, visited);
  }
  return true;
}

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) output[i] = distances_[i];
  return true;
}