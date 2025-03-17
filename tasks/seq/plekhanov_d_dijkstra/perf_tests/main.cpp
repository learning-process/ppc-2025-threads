#include <gtest/gtest.h>

#include <boost/graph/adjacency_list.hpp>  // NOLINT(misc-include-cleaner)
#include <boost/graph/dijkstra_shortest_paths.hpp>  // NOLINT(misc-include-cleaner)
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/plekhanov_d_dijkstra/include/ops_seq.hpp"

namespace plekhanov_d_dijkstra_seq {

std::vector<int> CalculateExpectedResult(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list,
                                         size_t start_vertex) {
  using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
                                      boost::property<boost::edge_weight_t, int>>;  // NOLINT(misc-include-cleaner)
  Graph graph(adj_list.size());

  for (size_t i = 0; i < adj_list.size(); ++i) {
    for (const auto &edge : adj_list[i]) {
      boost::add_edge(i, edge.first, edge.second, graph);  // NOLINT(misc-include-cleaner)
    }
  }

  std::vector<int> distances(boost::num_vertices(graph), INT_MAX);
  boost::dijkstra_shortest_paths(graph, start_vertex,
                                 boost::distance_map(distances.data()));  // NOLINT(misc-include-cleaner)
  return distances;
}

}  // namespace plekhanov_d_dijkstra_seq

TEST(plekhanov_d_dijkstra_seq, test_pipeline_run) {
  constexpr size_t kNumVertices = 6000;
  size_t start_vertex = 0;

  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  for (size_t i = 0; i < kNumVertices; ++i) {
    for (size_t j = 0; j < kNumVertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }

  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  auto test_task_sequential = std::make_shared<plekhanov_d_dijkstra_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<int> expected = plekhanov_d_dijkstra_seq::CalculateExpectedResult(adj_list, start_vertex);
  EXPECT_EQ(distances, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_task_run) {
  constexpr size_t kNumVertices = 6000;
  size_t start_vertex = 0;

  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  for (size_t i = 0; i < kNumVertices; ++i) {
    for (size_t j = 0; j < kNumVertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }

  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  auto test_task_sequential = std::make_shared<plekhanov_d_dijkstra_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<int> expected = plekhanov_d_dijkstra_seq::CalculateExpectedResult(adj_list, start_vertex);
  EXPECT_EQ(distances, expected);
}