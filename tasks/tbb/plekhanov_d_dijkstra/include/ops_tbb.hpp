#pragma once

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_dijkstra_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> graph_data_;
  std::vector<int> distances_;
  size_t start_vertex_;
  size_t num_vertices_;
  static const int kEndOfVertexList;

  std::vector<size_t> computeOffsets();
  void relaxEdges(int u, const std::vector<size_t>& offsets, std::vector<std::atomic<int>>& distances_atomic,
                  oneapi::tbb::concurrent_vector<int>& next_frontier);
};

}  // namespace plekhanov_d_dijkstra_tbb