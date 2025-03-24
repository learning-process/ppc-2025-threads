#include "omp/plekhanov_d_dijkstra/include/ops_omp.hpp"

#include <omp.h>

#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

const int plekhanov_d_dijkstra_omp::TestTaskOpenMP::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);
  distances_.resize(num_vertices_);

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  std::vector<std::vector<std::pair<int, int>>> graph(num_vertices_);
  size_t current_vertex = 0;
  size_t i = 0;

  while (i < graph_data_.size() && current_vertex < num_vertices_) {
    if (graph_data_[i] == kEndOfVertexList) {
      current_vertex++;
      i++;
      continue;
    }
    if (i + 1 >= graph_data_.size()) {
      break;
    }

    size_t dest = graph_data_[i];
    int weight = graph_data_[i + 1];
    if (weight < 0) {
      return false;
    }

    if (dest < num_vertices_) {
      graph[current_vertex].emplace_back(static_cast<int>(dest), weight);
    }
    i += 2;
  }

  std::vector<bool> visited(num_vertices_, false);

  for (int count = 0; count < static_cast<int>(num_vertices_) - 1; ++count) {
    int u = -1;
    int min_dist = INT_MAX;

#pragma omp parallel for
    for (int v = 0; v < static_cast<int>(num_vertices_); ++v) {
      if (!visited[v]) {
#pragma omp critical
        {
          if (distances_[v] < min_dist) {
            min_dist = distances_[v];
            u = v;
          }
        }
      }
    }

    if (u == -1) {
      break;
    }

    visited[u] = true;

#pragma omp parallel for
    for (int j = 0; j < static_cast<int>(graph[u].size()); ++j) {
      int v = graph[u][j].first;
      int weight = graph[u][j].second;
      if (!visited[v] && distances_[u] != INT_MAX) {
        int new_dist = distances_[u] + weight;
#pragma omp critical
        {
          distances_[v] = std::min(new_dist, distances_[v]);
        }
      }
    }
  }
  return true;
}

bool plekhanov_d_dijkstra_omp::TestTaskOpenMP::PostProcessingImpl() {
  auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}
