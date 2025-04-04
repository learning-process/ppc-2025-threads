#include <gtest/gtest.h>
#include <omp.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/zinoviev_a_convex_hull_components/include/ops_omp.hpp"

TEST(zinoviev_a_convex_hull_omp_perf, pipeline_run) {
  const int size = 1000;
  std::vector<int> input(size * size, 1);

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.push_back(size);
  data->inputs_count.push_back(size);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[4]));
  data->outputs_count.push_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_omp::ConvexHullOMP>(data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] { return omp_get_wtime(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->pipeline_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
}

TEST(zinoviev_a_convex_hull_omp_perf, task_run) {
  const int size = 1000;
  std::vector<int> input(size * size, 1);

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.push_back(size);
  data->inputs_count.push_back(size);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[4]));
  data->outputs_count.push_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_omp::ConvexHullOMP>(data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] { return omp_get_wtime(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->task_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
}