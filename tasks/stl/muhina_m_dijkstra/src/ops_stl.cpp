#include "stl/muhina_m_dijkstra/include/ops_stl.hpp"

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

const int muhina_m_dijkstra_stl::TestTaskSTL::kEndOfVertexList = -1;

namespace {
void RunDijkstraAlgorithm(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list, std::vector<int> &distances,
                          size_t start_vertex, int num_threads) {
  std::priority_queue<std::pair<int, size_t>, std::vector<std::pair<int, size_t>>, std::greater<>> pq;
  pq.push({0, start_vertex});
  std::mutex mtx;
  std::atomic<bool> finished(false);
  std::atomic<int> active_threads(0);

  auto worker = [&]() {
    while (true) {
      std::pair<int, size_t> current;
      bool has_work = false;

      {
        std::lock_guard<std::mutex> lock(mtx);
        if (pq.empty()) {
          if (finished) return;
          if (active_threads == 0) {
            finished = true;
            return;
          }
          continue;
        }
        current = pq.top();
        pq.pop();
        has_work = true;
        active_threads++;
      }

      if (!has_work) continue;

      size_t u = current.second;
      int dist_u = current.first;

      bool should_process = false;
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (dist_u <= distances[u]) {
          should_process = true;
        }
      }

      if (!should_process) {
        active_threads--;
        continue;
      }

      for (const auto &neighbor : adj_list[u]) {
        size_t v = neighbor.first;
        int weight = neighbor.second;
        int new_dist = dist_u + weight;

        bool updated = false;
        {
          std::lock_guard<std::mutex> lock(mtx);
          if (new_dist < distances[v]) {
            distances[v] = new_dist;
            updated = true;
          }
        }

        if (updated) {
          std::lock_guard<std::mutex> lock(mtx);
          pq.push({new_dist, v});
        }
      }

      active_threads--;
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  for (auto &t : threads) {
    t.join();
  }
}
}  // namespace

bool muhina_m_dijkstra_stl::TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);

  num_vertices_ = task_data->outputs_count[0];
  distances_.resize(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = INT_MAX;
  }
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;

  return true;
}

bool muhina_m_dijkstra_stl::TestTaskSTL::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool muhina_m_dijkstra_stl::TestTaskSTL::RunImpl() {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices_);
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
      adj_list[current_vertex].emplace_back(dest, weight);
    }

    i += 2;
  }
  const int num_threads = ppc::util::GetPPCNumThreads();
  RunDijkstraAlgorithm(adj_list, distances_, start_vertex_, num_threads);

  return true;
}

bool muhina_m_dijkstra_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < distances_.size(); ++i) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = distances_[i];
  }
  return true;
}
