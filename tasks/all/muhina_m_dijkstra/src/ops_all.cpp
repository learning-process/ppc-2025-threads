#include "all/muhina_m_dijkstra/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_priority_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/spin_mutex.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/datatype.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/serialization/access.hpp>
#include <climits>
#include <cstddef>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

struct MinVertex {
  int distance;
  int vertex;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & distance;
    ar & vertex;
  }
};

namespace boost {
namespace mpi {
template <>
struct is_mpi_datatype<MinVertex> : public mpl::true_ {};

template <>
struct minimum<MinVertex> {
  MinVertex operator()(const MinVertex& a, const MinVertex& b) const {
    if (b.distance < a.distance || (b.distance == a.distance && b.vertex < a.vertex)) {
      return b;
    }
    return a;
  }
};
}  // namespace mpi
}  // namespace boost

namespace {
void RunDijkstraAlgorithm(const std::vector<std::vector<std::pair<size_t, int>>>& adj_list, std::vector<int>& distances,
                          size_t start_vertex, boost::mpi::communicator& world, size_t num_vertices) {
  oneapi::tbb::concurrent_priority_queue<std::pair<int, int>, std::greater<>> pq;
  oneapi::tbb::spin_mutex mutex;

  std::vector<int> local_distances(num_vertices, INT_MAX);
  if (world.rank() == 0) {
    local_distances[start_vertex] = 0;
    pq.push({0, static_cast<int>(start_vertex)});
  }
  std::vector<int> global_distances(num_vertices);

  while (true) {
    int local_distance = INT_MAX;
    int local_vertex = -1;
    if (!pq.empty()) {
      std::pair<int, int> local_top;
      if (pq.try_pop(local_top)) {
        local_distance = local_top.first;
        local_vertex = local_top.second;
      }
    }

    MinVertex local_min = {local_distance, local_vertex};
    world.barrier();

    MinVertex global_min;
    boost::mpi::all_reduce(world, local_min, global_min, boost::mpi::minimum<MinVertex>());

    if (global_min.distance == INT_MAX) {
      break;
    }

    size_t u = static_cast<size_t>(global_min.vertex);
    int dist_u = global_min.distance;

    if (u >= num_vertices) {
      break;
    }

    if (dist_u > local_distances[u]) {
      continue;
    }

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, adj_list[u].size()), [&](const oneapi::tbb::blocked_range<size_t>& r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            size_t v = adj_list[u][i].first;
            int weight = adj_list[u][i].second;
            int new_dist = (local_distances[u] == INT_MAX || weight == INT_MAX) ? INT_MAX : local_distances[u] + weight;

            if (new_dist < local_distances[v]) {
              oneapi::tbb::spin_mutex::scoped_lock lock(mutex);
              if (new_dist < local_distances[v]) {
                local_distances[v] = new_dist;
                pq.push({new_dist, static_cast<int>(v)});
              }
            }
          }
        });

    boost::mpi::all_reduce(world, local_distances.data(), num_vertices, global_distances.data(),
                           boost::mpi::minimum<int>());

    for (size_t i = 0; i < num_vertices; ++i) {
      local_distances[i] = global_distances[i];
    }
  }

  distances = local_distances;
}
}  // namespace

const int muhina_m_dijkstra_all::TestTaskALL::kEndOfVertexList = -1;

bool muhina_m_dijkstra_all::TestTaskALL::PreProcessingImpl() {
  graph_data_.clear();
  distances_.clear();

  world_.barrier();
  if (!task_data) {
    return false;
  }

  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    if (!in_ptr) {
      return false;
    }
    graph_data_.assign(in_ptr, in_ptr + input_size);
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, graph_data_);
    }
  } else {
    world_.recv(0, 0, graph_data_);
  }

  world_.barrier();

  if (task_data->outputs_count.empty()) {
    return false;
  }

  num_vertices_ = task_data->outputs_count[0];
  boost::mpi::broadcast(world_, num_vertices_, 0);
  world_.barrier();

  if (num_vertices_ == 0 || num_vertices_ > 10000) {
    return false;
  }

  distances_.resize(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = INT_MAX;
  }
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    if (world_.rank() == 0) {
      start_vertex_ = *reinterpret_cast<int*>(task_data->inputs[1]);
    }
    boost::mpi::broadcast(world_, start_vertex_, 0);
  } else {
    start_vertex_ = 0;
  }

  return true;
}

bool muhina_m_dijkstra_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0;
    return !task_data->outputs_count.empty() && task_data->outputs_count[0] > 0;
  }
  return true;
}

bool muhina_m_dijkstra_all::TestTaskALL::RunImpl() {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices_);
  size_t current_vertex = 0;
  size_t i = 0;
  world_.barrier();
  while (i < graph_data_.size() && current_vertex < num_vertices_) {
    if (i >= graph_data_.size()) {
      return false;
    }
    if (graph_data_[i] == kEndOfVertexList) {
      current_vertex++;
      i++;
      continue;
    }
    if (i + 1 >= graph_data_.size()) {
      break;
    }

    size_t dest = static_cast<size_t>(graph_data_[i]);
    int weight = graph_data_[i + 1];
    if (weight < 0) {
      return false;
    }

    if (dest < num_vertices_) {
      adj_list[current_vertex].push_back({dest, weight});
    }
    i += 2;
  }
  RunDijkstraAlgorithm(adj_list, distances_, start_vertex_, world_, num_vertices_);
  return true;
}

bool muhina_m_dijkstra_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < distances_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = distances_[i];
    }
  }
  return true;
}
