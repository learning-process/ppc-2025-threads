#include "stl/plekhanov_d_dijkstra/include/ops_stl.hpp"

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

const int plekhanov_d_dijkstra_stl::TestTaskSTL::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_stl::TestTaskSTL::PreProcessingImpl() {
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

bool plekhanov_d_dijkstra_stl::TestTaskSTL::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->inputs_count[0] == 0 || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool plekhanov_d_dijkstra_stl::TestTaskSTL::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  std::vector<std::vector<std::pair<int, int>>> graph(num_vertices_);
  size_t current_vertex = 0;
  int i = 0;

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
  std::mutex mtx;

  for (int count = 0; count < static_cast<int>(num_vertices_) - 1; ++count) {
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<int> local_mins(num_threads, std::numeric_limits<int>::max());
    std::vector<int> local_verts(num_threads, -1);

    auto find_min = [&](int thread_id, int start, int end) {
      int& local_min = local_mins[thread_id];
      int& current_u = local_verts[thread_id];
      for (int v = start; v < end; ++v) {
        if (!visited[v] && distances_[v] < local_min) {
          local_min = distances_[v];
          current_u = v;
        }
      }
    };

    std::vector<std::thread> threads;
    const int chunk_size = num_vertices_ / num_threads;
    for (i = 0; i < num_threads; ++i) {
      int start = i * chunk_size;
      int end = (i == num_threads - 1) ? num_vertices_ : (i + 1) * chunk_size;
      threads.emplace_back(find_min, i, start, end);
    }

    for (auto& t : threads) t.join();

    int u = -1;
    int global_min = std::numeric_limits<int>::max();
    for (i = 0; i < num_threads; ++i) {
      if (local_mins[i] < global_min && local_verts[i] != -1) {
        global_min = local_mins[i];
        u = local_verts[i];
      }
    }

    if (u == -1) break;
    visited[u] = true;

    auto process_edges = [&](int start, int end) {
      for (int j = start; j < end; ++j) {
        int v = graph[u][j].first;
        int weight = graph[u][j].second;
        if (!visited[v] && distances_[u] != std::numeric_limits<int>::max()) {
          int new_dist = distances_[u] + weight;
          std::lock_guard<std::mutex> lock(mtx);
          if (new_dist < distances_[v]) {
            distances_[v] = new_dist;
          }
        }
      }
    };

    const int edges_count = static_cast<int>(graph[u].size());
    const int edge_chunk = std::max(1, edges_count / num_threads);
    threads.clear();

    for (i = 0; i < num_threads; ++i) {
      int start = i * edge_chunk;
      int end = (i == num_threads - 1) ? edges_count : (i + 1) * edge_chunk;
      if (start >= edges_count) break;
      threads.emplace_back(process_edges, start, end);
    }

    for (auto& t : threads) t.join();
  }
  return true;
}

bool plekhanov_d_dijkstra_stl::TestTaskSTL::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}
