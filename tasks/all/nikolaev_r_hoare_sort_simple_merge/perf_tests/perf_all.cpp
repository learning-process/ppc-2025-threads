#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GenerateRandomVector(size_t len, double min_val = -1000.0, double max_val = 1000.0) {
  std::vector<double> vect(len);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_val, max_val);

  for (size_t i = 0; i < len; ++i) {
    vect[i] = dis(gen);
  }

  return vect;
}

}  // namespace

TEST(nikolaev_r_hoare_sort_simple_merge_all, test_pipeline_run) {
  boost::mpi::communicator world;
  constexpr size_t kLen = 500000;

  std::vector<double> in = GenerateRandomVector(kLen);
  std::vector<double> out(kLen, 0.0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(nikolaev_r_hoare_sort_simple_merge_all, test_task_run) {
  boost::mpi::communicator world;
  constexpr size_t kLen = 500000;

  std::vector<double> in = GenerateRandomVector(kLen);
  std::vector<double> out(kLen, 0.0);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
