#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(Konstantinov_I_Sort_Batcher_all, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  unsigned int seed = world.rank() == 0 ? std::random_device{}() : 0;
  boost::mpi::broadcast(world, seed, 0);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  const int kCount = 100000;
  std::vector<double> in, exp_out;

  if (world.rank() == 0) {
    in.resize(kCount);
    exp_out.resize(kCount);
    for (int i = 0; i < kCount; i++) {
      in[i] = dist(gen);
      exp_out[i] = in[i];
    }
    std::sort(exp_out.begin(), exp_out.end());
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->outputs_count.emplace_back(in.size());
  } else {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(0);
    task_data->outputs.emplace_back(nullptr);
    task_data->outputs_count.emplace_back(0);
  }

  auto test_task = std::make_shared<konstantinov_i_sort_batcher_all::RadixSortBatcherall>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

  if (world.rank() == 0) {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  } else {
    test_task->Run();
  }

  if (world.rank() == 0) {
    for (size_t i = 1; i < in.size(); ++i) {
      ASSERT_LE(in[i - 1], in[i]) << "Vector not sorted at position " << i;
    }
  }

  world.barrier();
}

TEST(Konstantinov_I_Sort_Batcher_all, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  unsigned int seed = world.rank() == 0 ? std::random_device{}() : 0;
  boost::mpi::broadcast(world, seed, 0);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  const int kCount = 100000;
  std::vector<double> in, exp_out;

  if (world.rank() == 0) {
    in.resize(kCount);
    exp_out.resize(kCount);
    for (int i = 0; i < kCount; i++) {
      in[i] = dist(gen);
      exp_out[i] = in[i];
    }
    std::sort(exp_out.begin(), exp_out.end());
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));  // In-place sort
    task_data->outputs_count.emplace_back(in.size());
  } else {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(0);
    task_data->outputs.emplace_back(nullptr);
    task_data->outputs_count.emplace_back(0);
  }

  auto test_task = std::make_shared<konstantinov_i_sort_batcher_all::RadixSortBatcherall>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

  if (world.rank() == 0) {
    perf_analyzer->TaskRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  } else {
    test_task->Run();
  }

  if (world.rank() == 0) {
    for (size_t i = 1; i < in.size(); ++i) {
      ASSERT_LE(in[i - 1], in[i]) << "Vector not sorted at position " << i;
    }
  }

  world.barrier();
}