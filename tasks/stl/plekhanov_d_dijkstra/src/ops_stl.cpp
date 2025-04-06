#include "stl/plekhanov_d_dijkstra/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

const int plekhanov_d_dijkstra_stl::TestTaskSTL::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_stl::TestTaskSTL::PreProcessingImpl() {
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

bool plekhanov_d_dijkstra_stl::TestTaskSTL::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_stl::TestTaskSTL::RunImpl() {  // NOLINT
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
  std::vector<std::atomic<int>> distances_atomic(num_vertices_);
  for (auto& d : distances_atomic) {
    d.store(INT_MAX);
  }
  distances_atomic[start_vertex_] = 0;

  size_t num_threads = std::min(static_cast<size_t>(ppc::util::GetPPCNumThreads()),
                                static_cast<size_t>(std::thread::hardware_concurrency()));

  auto find_min_vertex_parallel = [&](int& min_vertex) {
    std::mutex mutex;
    int local_min_dist = INT_MAX;
    min_vertex = -1;

    size_t chunk_size = (num_vertices_ + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t start = t * chunk_size;
      size_t end = std::min(start + chunk_size, num_vertices_);

      threads.emplace_back([&, start, end]() {
        int thread_min = -1;
        int thread_min_dist = INT_MAX;

        for (size_t i = start; i < end; ++i) {
          if (!visited[i]) {
            int d = distances_atomic[i].load();
            if (d < thread_min_dist) {
              thread_min_dist = d;
              thread_min = static_cast<int>(i);
            }
          }
        }

        if (thread_min != -1) {
          std::lock_guard<std::mutex> lock(mutex);
          if (thread_min_dist < local_min_dist) {
            local_min_dist = thread_min_dist;
            min_vertex = thread_min;
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  };

  for (size_t count = 0; count < num_vertices_; ++count) {
    int u = -1;
    find_min_vertex_parallel(u);
    if (u == -1 || distances_atomic[u] == INT_MAX) {
      break;
    }

    visited[u] = true;

    const auto& neighbors = graph[u];
    size_t edge_count = neighbors.size();
    size_t chunk_size = (edge_count + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t start = t * chunk_size;
      size_t end = std::min(start + chunk_size, edge_count);

      threads.emplace_back([&, start, end]() {
        for (size_t i = start; i < end; ++i) {
          int v = neighbors[i].first;
          int weight = neighbors[i].second;

          int cur_dist = distances_atomic[u].load();
          int new_dist = cur_dist + weight;

          int old_val = distances_atomic[v].load();
          while (new_dist < old_val && !distances_atomic[v].compare_exchange_weak(old_val, new_dist)) {
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  distances_.resize(num_vertices_);
  for (size_t k = 0; k < num_vertices_; ++k) {
    distances_[k] = distances_atomic[k].load();
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