#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace trubin_a_algorithm_dijkstra_stl {

struct Edge {
  size_t to;
  int weight;

  Edge(size_t to, int weight) : to(to), weight(weight) {}
};

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  bool BuildAdjacencyList(const std::vector<int>& graph_data);

  using QueueElement = std::pair<int, size_t>;

  void ProcessEdge(size_t from, const Edge& edge,
                   std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap,
                   std::mutex& distances_mutex, std::mutex& heap_mutex);

  void ProcessTaskBlock(const std::vector<std::pair<size_t, size_t>>& block,
                        std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap,
                        std::mutex& distances_mutex, std::mutex& heap_mutex);

  void ThreadWorker(std::queue<std::vector<std::pair<size_t, size_t>>>& task_queue, std::mutex& task_mutex,
                    std::condition_variable& task_cv, bool& task_done,
                    std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap,
                    std::mutex& distances_mutex, std::mutex& heap_mutex);

  bool ProcessNextVertex(std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>>& min_heap,
                         std::mutex& heap_mutex, std::mutex& distances_mutex,
                         std::queue<std::vector<std::pair<size_t, size_t>>>& task_queue, std::mutex& task_mutex,
                         std::condition_variable& task_cv, size_t num_threads);

  std::vector<std::vector<Edge>> adjacency_list_;
  std::vector<int> distances_;
  size_t start_vertex_ = 0;
  size_t num_vertices_ = 0;

  bool validation_passed_ = false;

  static constexpr int kEndOfVertexList = -1;
};

}  // namespace trubin_a_algorithm_dijkstra_stl
