#include "tbb/plekhanov_d_dijkstra/include/ops_tbb.hpp"

#include <atomic>
#include <climits>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include <oneapi/tbb/task_arena.h>
#include "tbb/tbb.h"

const int plekhanov_d_dijkstra_tbb::TestTaskTBB::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.resize(num_vertices_);
  distances_.assign(num_vertices_, INT_MAX);
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::RunImpl() {

  std::vector<std::atomic<int>> distances_atomic(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_atomic[i].store(distances_[i], std::memory_order_relaxed);
  }
  std::vector<int> frontier;
  frontier.push_back(start_vertex_);

  std::vector<size_t> offsets(num_vertices_ + 1, 0);
  {
    size_t pos = 0;
    for (size_t vertex = 0; vertex < num_vertices_; ++vertex) {
      offsets[vertex] = pos;
      while (pos < graph_data_.size() && graph_data_[pos] != kEndOfVertexList) {
        pos += 2;
      }
      if (pos < graph_data_.size() && graph_data_[pos] == kEndOfVertexList) {
        ++pos;
      }
    }
    offsets[num_vertices_] = pos;
  }

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    while (!frontier.empty()) {
      oneapi::tbb::concurrent_vector<int> next_frontier;

      oneapi::tbb::parallel_for(static_cast<size_t>(0), frontier.size(), [&](size_t i) {
        int u = frontier[i];
        size_t begin = offsets[u];
        size_t end = offsets[u + 1];
        for (size_t pos = begin; pos < end;) {
          int v = graph_data_[pos++];
          int weight = graph_data_[pos++];
          int cur_dist = distances_atomic[u].load(std::memory_order_relaxed);
          int new_dist = cur_dist + weight;
          int old_dist = distances_atomic[v].load(std::memory_order_relaxed);
          while (new_dist < old_dist) {
            if (distances_atomic[v].compare_exchange_weak(old_dist, new_dist, std::memory_order_relaxed)) {
              next_frontier.push_back(v);
              break;
            }
          }
        }
      });

      frontier.clear();
      for (auto v : next_frontier) {
        frontier.push_back(v);
      }
    }
  });

  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = distances_atomic[i].load(std::memory_order_relaxed);
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
