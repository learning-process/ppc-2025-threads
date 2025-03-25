#include "tbb/plekhanov_d_dijkstra/include/ops_tbb.hpp"

#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "tbb/tbb.h"

const int plekhanov_d_dijkstra_tbb::TestTaskTBB::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.assign(num_vertices_, INT_MAX);
  distances_.resize(num_vertices_);

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->inputs_count[0] == 0 || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::RunImpl() {
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
    std::atomic<int> u = -1;
    std::atomic<int> min_dist = std::numeric_limits<int>::max();

    oneapi::tbb::parallel_for(0, static_cast<int>(num_vertices_), [&](int v) {
      if (!visited[v] && distances_[v] < min_dist) {
        min_dist = distances_[v];
        u = v;
      }
    });

    if (u == -1) {
      break;
    }

    visited[u] = true;

    oneapi::tbb::parallel_for(0, static_cast<int>(graph[u].size()), [&](int j) {
      int v = graph[u][j].first;
      int weight = graph[u][j].second;
      if (!visited[v] && distances_[u] != std::numeric_limits<int>::max()) {
        int new_dist = distances_[u] + weight;
        oneapi::tbb::mutex::scoped_lock lock(mutex_);
        distances_[v] = std::min(new_dist, distances_[v]);
      }
    });
  }
  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}
