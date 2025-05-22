#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/oturin_a_gift_wrapping/include/ops_mpi.hpp"

namespace {
oturin_a_gift_wrapping_mpi::Coord RandCoord(int r) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(-r, r);
  return {.x = dist(rng), .y = dist(rng)};
}
}  // namespace

TEST(oturin_a_gift_wrapping_mpi, test_pipeline_run) {
  int count = 250000;
  using namespace oturin_a_gift_wrapping_mpi;

  // Create data
  std::vector<Coord> in(count);
  std::vector<Coord> out(4);
  std::vector<Coord> ans = {{.x = -5, .y = 5}, {.x = 5, .y = 5}, {.x = 5, .y = -5}, {.x = -5, .y = -5}};
  auto gen = [&]() { return RandCoord(4); };
  std::ranges::generate(in.begin(), in.end(), gen);
  in.insert(in.end(), ans.begin(), ans.end());

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<oturin_a_gift_wrapping_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  for (std::size_t i = 0; i < ans.size(); i++) {
    EXPECT_EQ(ans[i].x, out[i].x);
    EXPECT_EQ(ans[i].y, out[i].y);
  }
}

TEST(oturin_a_gift_wrapping_mpi, test_task_run) {
  int count = 250000;
  using namespace oturin_a_gift_wrapping_mpi;

  // Create data
  std::vector<Coord> in(count);
  std::vector<Coord> out(4);
  std::vector<Coord> ans = {{.x = -5, .y = 5}, {.x = 5, .y = 5}, {.x = 5, .y = -5}, {.x = -5, .y = -5}};
  auto gen = [&]() { return RandCoord(4); };
  std::ranges::generate(in.begin(), in.end(), gen);
  in.insert(in.end(), ans.begin(), ans.end());

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<oturin_a_gift_wrapping_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  for (std::size_t i = 0; i < ans.size(); i++) {
    EXPECT_EQ(ans[i].x, out[i].x);
    EXPECT_EQ(ans[i].y, out[i].y);
  }
}