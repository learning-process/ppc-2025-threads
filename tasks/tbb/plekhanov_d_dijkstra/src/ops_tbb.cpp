#include "tbb/plekhanov_d_dijkstra/include/ops_tbb.hpp"

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/task_arena.h>

#include <atomic>
#include <climits>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "tbb/tbb.h"

const int plekhanov_d_dijkstra_tbb::TestTaskTBB::kEndOfVertexList = -1;

namespace plekhanov_d_dijkstra_tbb {

std::vector<size_t> TestTaskTBB::computeOffsets() {
  std::vector<size_t> offsets(num_vertices_ + 1, 0);
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
  return offsets;
}

void TestTaskTBB::relaxEdges(int u, const std::vector<size_t>& offsets, std::vector<std::atomic<int>>& distances_atomic,
                             oneapi::tbb::concurrent_vector<int>& next_frontier) {
  size_t begin = offsets[u];
  size_t end = offsets[u + 1];
  int cur_dist = distances_atomic[u].load(std::memory_order_relaxed);
  for (size_t pos = begin; pos < end;) {
    int v = graph_data_[pos++];
    int weight = graph_data_[pos++];
    int new_dist = cur_dist + weight;
    int old_dist = distances_atomic[v].load(std::memory_order_relaxed);
    while (new_dist < old_dist) {
      if (distances_atomic[v].compare_exchange_weak(old_dist, new_dist, std::memory_order_relaxed)) {
        next_frontier.push_back(v);
        break;
      }
    }
  }
}

}  // namespace plekhanov_d_dijkstra_tbb

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
  if (task_data->inputs_count.empty() || task_data->inputs_count[0] == 0 || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] == 0) {
    return false;
  }

  for (size_t pos = 0; pos < graph_data_.size();) {
    if (graph_data_[pos] == kEndOfVertexList) {
      ++pos;
      continue;
    }
    if (pos + 1 < graph_data_.size()) {
      int weight = graph_data_[pos + 1];
      if (weight < 0) {
        return false;
      }
    }
    pos += 2;
  }

  return true;
}

bool plekhanov_d_dijkstra_tbb::TestTaskTBB::RunImpl() {
  std::vector<std::atomic<int>> distances_atomic(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_atomic[i].store(distances_[i], std::memory_order_relaxed);
  }

  std::vector<int> frontier{static_cast<int>(start_vertex_)};

  const auto offsets = computeOffsets();

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    while (!frontier.empty()) {
      oneapi::tbb::concurrent_vector<int> next_frontier;
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, frontier.size()),
                                [&](const oneapi::tbb::blocked_range<size_t>& range) {
                                  for (size_t i = range.begin(); i != range.end(); ++i) {
                                    relaxEdges(frontier[i], offsets, distances_atomic, next_frontier);
                                  }
                                });
      frontier.assign(next_frontier.begin(), next_frontier.end());
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
