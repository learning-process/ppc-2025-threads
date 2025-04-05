#include "stl/plekhanov_d_dijkstra/include/ops_stl.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
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

bool plekhanov_d_dijkstra_stl::TestTaskSTL::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
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

  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
  pq.emplace(0, start_vertex_);
  distances_[start_vertex_] = 0;

  std::mutex pq_mutex;

  const size_t num_threads =
      std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  while (!pq.empty()) {
    pq_mutex.lock();
    auto top_element = pq.top();
    int dist = top_element.first;
    int u = top_element.second;
    pq.pop();
    pq_mutex.unlock();

    if (dist > distances_[u]) {
      continue;
    }

    size_t edges_count = graph[u].size();
    size_t chunk_size = (edges_count + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;

    for (unsigned int j = 0; j < num_threads; ++j) {
      size_t start = j * chunk_size;
      size_t end = std::min(start + chunk_size, edges_count);

      futures.push_back(std::async(std::launch::async, [&, start, end]() {
        for (size_t k = start; k < end; ++k) {
          int v = graph[u][k].first;
          int weight = graph[u][k].second;
          int new_dist = dist + weight;

          if (new_dist < distances_[v]) {
            distances_[v] = new_dist;
            pq_mutex.lock();
            pq.emplace(new_dist, v);
            pq_mutex.unlock();
          }
        }
      }));
    }

    for (auto& f : futures) {
      f.get();
    }
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