#include <gtest/gtest.h>
#include <omp.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/zinoviev_a_convex_hull_components/include/ops_omp.hpp"

TEST(zinoviev_a_convex_hull_omp_perf, PipelineRun) {
  const int SIZE = 1000;
  std::vector<int> input(SIZE * SIZE, 1);

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.push_back(SIZE);
  data->inputs_count.push_back(SIZE);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[4]));
  data->outputs_count.push_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_omp::ConvexHullOMP>(data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
}

TEST(zinoviev_a_convex_hull_omp_perf, TaskRun) {
  const int SIZE = 1000;
  std::vector<int> input(SIZE * SIZE, 1);

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  data->inputs_count.push_back(SIZE);
  data->inputs_count.push_back(SIZE);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_omp::Point[4]));
  data->outputs_count.push_back(4);

  auto task = std::make_shared<zinoviev_a_convex_hull_components_omp::ConvexHullOMP>(data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);

  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_omp::Point*>(data->outputs[0]);
}