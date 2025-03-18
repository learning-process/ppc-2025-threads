#include <gtest/gtest.h>

#include <boost/graph/adjacency_list.hpp>           // NOLINT(misc-include-cleaner)
#include <boost/graph/dijkstra_shortest_paths.hpp>  // NOLINT(misc-include-cleaner)
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/plekhanov_d_dijkstra/include/ops_seq.hpp"

namespace plekhanov_d_dijkstra_seq {

void static RunValidationFailureTest();  // NOLINT(misc-use-anonymous-namespace)

template <typename ExpectedResultType>
void RunTest(const std::vector<std::vector<std::pair<size_t, int>>> &adj_list, size_t start_vertex,
             const std::vector<ExpectedResultType> &expected_result, bool expect_success = true) {
  const size_t k_num_vertices = adj_list.size();
  std::vector<int> distances(k_num_vertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_seq->outputs_count.emplace_back(k_num_vertices);
  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  if (expect_success) {
    ASSERT_TRUE(test_task_sequential.Run());
    test_task_sequential.PostProcessing();
    for (size_t i = 0; i < k_num_vertices; ++i) {
      EXPECT_EQ(distances[i], expected_result[i]);
    }
  } else {
    ASSERT_FALSE(test_task_sequential.Run());
    test_task_sequential.PostProcessing();
  }
}

void static RunValidationFailureTest() {
  std::vector<int> graph_data;
  size_t start_vertex = 0;
  size_t num_vertices = 0;
  std::vector<int> distances(num_vertices, INT_MAX);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_seq->outputs_count.emplace_back(num_vertices);
  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.Validation());
}

std::vector<std::vector<std::pair<size_t, int>>> static GenerateRandomGraph(  // NOLINT(misc-use-anonymous-namespace)
    size_t num_vertices) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 10);

  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = i + 1; j < num_vertices; ++j) {
      if (gen() % 2 == 0) {
        adj_list[i].emplace_back(j, dis(gen));
        adj_list[j].emplace_back(i, dis(gen));
      }
    }
  }
  return adj_list;
}

std::vector<int> static CalculateExpectedResult(                       // NOLINT(misc-use-anonymous-namespace)
    const std::vector<std::vector<std::pair<size_t, int>>> &adj_list,  // NOLINT(misc-use-anonymous-namespace)
    size_t start_vertex) {
  using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,   // NOLINT(misc-include-cleaner)
                                      boost::no_property,                           // NOLINT(misc-include-cleaner)
                                      boost::property<boost::edge_weight_t, int>>;  // NOLINT(misc-include-cleaner)
  Graph graph(adj_list.size());                                                     // NOLINT(misc-include-cleaner)

  for (size_t i = 0; i < adj_list.size(); ++i) {
    for (const auto &edge : adj_list[i]) {
      boost::add_edge(i, edge.first, edge.second, graph);  // NOLINT(misc-include-cleaner)
    }
  }

  std::vector<int> distances(boost::num_vertices(graph), INT_MAX);  // NOLINT(misc-include-cleaner)
  boost::dijkstra_shortest_paths(graph, start_vertex,
                                 boost::distance_map(distances.data()));  // NOLINT(misc-include-cleaner)
  return distances;
}

}  // namespace plekhanov_d_dijkstra_seq

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Simple_Path_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {
      {{1, 1}}, {{0, 1}, {2, 2}}, {{1, 2}, {3, 3}}, {{2, 3}, {4, 4}}, {{3, 4}}};
  std::vector<int> expected = {0, 1, 3, 6, 10};
  plekhanov_d_dijkstra_seq::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Complete_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 10}, {2, 3}, {3, 20}, {4, 7}},
                                                               {{0, 10}, {2, 5}, {3, 4}, {4, 11}},
                                                               {{0, 3}, {1, 5}, {3, 2}, {4, 6}},
                                                               {{0, 20}, {1, 4}, {2, 2}, {4, 8}},
                                                               {{0, 7}, {1, 11}, {2, 6}, {3, 8}}};
  std::vector<int> expected = {3, 5, 0, 2, 6};
  plekhanov_d_dijkstra_seq::RunTest(adj_list, 2, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Disconnected_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 5}, {2, 3}}, {{0, 5}, {2, 1}}, {{0, 3}, {1, 1}},
                                                               {{4, 2}, {5, 8}}, {{3, 2}, {5, 1}}, {{3, 8}, {4, 1}}};
  std::vector<int> expected = {0, 4, 3, INT_MAX, INT_MAX, INT_MAX};
  plekhanov_d_dijkstra_seq::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Large_Sparse_Graph) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {
      {{1, 4}, {2, 2}}, {{3, 5}, {4, 10}}, {{5, 3}, {6, 2}}, {{7, 4}}, {{8, 11}},
      {{8, 1}},         {{9, 3}},          {{9, 5}},         {{9, 7}}, {}};

  std::vector<int> expected = {0, 4, 2, 9, 14, 5, 4, 13, 6, 7};
  plekhanov_d_dijkstra_seq::RunTest(adj_list, 0, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_validation_failure) {
  plekhanov_d_dijkstra_seq::RunValidationFailureTest();
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Negative_Edges) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list = {{{1, 4}, {2, -2}}, {{0, 4}, {2, 3}}, {{0, -2}, {1, 3}}};
  std::vector<int> expected = {0, 0, 0};
  plekhanov_d_dijkstra_seq::RunTest(adj_list, 0, expected, false);
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Random_Graph_10) {
  size_t num_vertices = 10;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list =
      plekhanov_d_dijkstra_seq::GenerateRandomGraph(num_vertices);
  size_t start_vertex = 0;

  std::vector<int> expected = plekhanov_d_dijkstra_seq::CalculateExpectedResult(adj_list, start_vertex);

  plekhanov_d_dijkstra_seq::RunTest(adj_list, start_vertex, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_dijkstra_Random_Graph_30) {
  size_t num_vertices = 30;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list =
      plekhanov_d_dijkstra_seq::GenerateRandomGraph(num_vertices);
  size_t start_vertex = 0;

  std::vector<int> expected = plekhanov_d_dijkstra_seq::CalculateExpectedResult(adj_list, start_vertex);

  plekhanov_d_dijkstra_seq::RunTest(adj_list, start_vertex, expected);
}