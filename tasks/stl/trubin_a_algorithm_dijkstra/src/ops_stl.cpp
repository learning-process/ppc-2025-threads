#include "stl/trubin_a_algorithm_dijkstra/include/ops_stl.hpp"

#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::PreProcessingImpl() {
  if (!validation_passed_) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::vector<int> graph_data(in_ptr, in_ptr + input_size);

  num_vertices_ = task_data->outputs_count[0];

  if (num_vertices_ == 0 || input_size == 0) {
    return true;
  }

  adjacency_list_.assign(num_vertices_, {});
  distances_.assign(num_vertices_, std::numeric_limits<int>::max());

  if (!BuildAdjacencyList(graph_data)) {
    return false;
  }

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    int* ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    if (*ptr < 0 || static_cast<size_t>(*ptr) >= num_vertices_) {
      return false;
    }
    start_vertex_ = static_cast<size_t>(*ptr);
  } else {
    start_vertex_ = 0;
  }

  distances_[start_vertex_] = 0;
  return true;
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  if (task_data->outputs_count[0] == 0 && task_data->inputs_count[0] == 0) {
    validation_passed_ = true;
    return true;
  }

  if (task_data->outputs[0] == nullptr || task_data->outputs_count[0] == 0) {
    return false;
  }

  validation_passed_ = true;
  return true;
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::RunImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (start_vertex_ >= num_vertices_) {
    return false;
  }

  using QueueElement = std::pair<int, size_t>;
  std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>> min_heap;
  min_heap.emplace(0, start_vertex_);

  size_t num_threads =
      std::min(static_cast<size_t>(ppc::util::GetPPCNumThreads()), num_vertices_ > 0 ? num_vertices_ : 1);
  if (num_threads > 8) {
    num_threads = 8;
  }

  std::mutex heap_mutex;
  std::mutex distances_mutex;

  std::queue<std::vector<std::pair<size_t, size_t>>> task_queue;
  std::mutex task_mutex;
  std::condition_variable task_cv;
  bool task_done = false;

  std::vector<std::thread> threads;
  for (size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        [&]() { ThreadWorker(task_queue, task_mutex, task_cv, task_done, min_heap, distances_mutex, heap_mutex); });
  }

  while (ProcessNextVertex(min_heap, heap_mutex, distances_mutex, task_queue, task_mutex, task_cv, num_threads)) {
  }

  {
    std::lock_guard<std::mutex> lock(task_mutex);
    task_done = true;
  }
  task_cv.notify_all();

  for (auto& th : threads) {
    th.join();
  }

  return true;
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::PostProcessingImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < num_vertices_; ++i) {
    int d = distances_[i];
    out_ptr[i] = (d == std::numeric_limits<int>::max()) ? -1 : d;
  }
  return true;
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::BuildAdjacencyList(const std::vector<int>& graph_data) {
  size_t current_vertex = 0;
  size_t i = 0;

  while (i < graph_data.size()) {
    if (graph_data[i] == kEndOfVertexList) {
      if (current_vertex >= num_vertices_) {
        return false;
      }
      current_vertex++;
      i++;
      continue;
    }

    if (i + 1 >= graph_data.size()) {
      return false;
    }

    auto to = static_cast<size_t>(graph_data[i]);
    int weight = graph_data[i + 1];

    if (to >= num_vertices_ || weight < 0) {
      return false;
    }

    adjacency_list_[current_vertex].emplace_back(to, weight);
    i += 2;
  }

  return true;
}

void trubin_a_algorithm_dijkstra_stl::TestTaskSTL::ProcessEdge(
    size_t from, const Edge& edge,
    std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap, std::mutex& distances_mutex,
    std::mutex& heap_mutex) {
  int new_dist;
  bool should_add = false;
  {
    std::lock_guard<std::mutex> lock(distances_mutex);
    new_dist = distances_[from] + edge.weight;
    if (new_dist < distances_[edge.to]) {
      distances_[edge.to] = new_dist;
      should_add = true;
    }
  }
  if (should_add) {
    std::lock_guard<std::mutex> lock(heap_mutex);
    min_heap.emplace(new_dist, edge.to);
  }
}

void trubin_a_algorithm_dijkstra_stl::TestTaskSTL::ProcessTaskBlock(
    const std::vector<std::pair<size_t, size_t>>& block,
    std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap, std::mutex& distances_mutex,
    std::mutex& heap_mutex) {
  for (const auto& [from, edge_idx] : block) {
    const auto& edge = adjacency_list_[from][edge_idx];
    ProcessEdge(from, edge, min_heap, distances_mutex, heap_mutex);
  }
}

void trubin_a_algorithm_dijkstra_stl::TestTaskSTL::ThreadWorker(
    std::queue<std::vector<std::pair<size_t, size_t>>>& task_queue, std::mutex& task_mutex,
    std::condition_variable& task_cv, bool& task_done,
    std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap, std::mutex& distances_mutex,
    std::mutex& heap_mutex) {
  while (true) {
    std::vector<std::pair<size_t, size_t>> block;
    {
      std::unique_lock<std::mutex> lock(task_mutex);
      task_cv.wait(lock, [&]() { return task_done || !task_queue.empty(); });
      if (task_done && task_queue.empty()) break;
      block = std::move(task_queue.front());
      task_queue.pop();
    }
    ProcessTaskBlock(block, min_heap, distances_mutex, heap_mutex);
  }
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::ProcessNextVertex(
    std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap, std::mutex& heap_mutex,
    std::mutex& distances_mutex, std::queue<std::vector<std::pair<size_t, size_t>>>& task_queue, std::mutex& task_mutex,
    std::condition_variable& task_cv, size_t num_threads) {
  size_t from_vertex = 0;
  {
    std::lock_guard<std::mutex> lock(heap_mutex);
    if (min_heap.empty()) {
      return false;
    }
    auto [dist, u] = min_heap.top();
    min_heap.pop();
    if (dist > distances_[u]) {
      return true;
    }
    from_vertex = u;
  }

  const auto& edges = adjacency_list_[from_vertex];
  size_t total_edges = edges.size();
  if (total_edges == 0) {
    return true;
  }

  if (total_edges < num_threads * 4) {
    for (size_t i = 0; i < total_edges; ++i) {
      const auto& edge = edges[i];
      ProcessEdge(from_vertex, edge, min_heap, distances_mutex, heap_mutex);
    }
  } else {
    size_t block_size = (total_edges + num_threads - 1) / num_threads;
    std::vector<std::vector<std::pair<size_t, size_t>>> tasks(num_threads);

    for (size_t i = 0; i < total_edges; ++i) {
      size_t block_id = i / block_size;
      if (block_id >= num_threads) block_id = num_threads - 1;
      tasks[block_id].emplace_back(from_vertex, i);
    }

    {
      std::lock_guard<std::mutex> lock(task_mutex);
      for (auto& task : tasks) {
        if (!task.empty()) {
          task_queue.emplace(std::move(task));
        }
      }
    }
    task_cv.notify_all();
  }

  return true;
}
