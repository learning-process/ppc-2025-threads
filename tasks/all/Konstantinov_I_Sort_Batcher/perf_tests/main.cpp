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

  constexpr int kCount = 10;
  std::vector<double> in, exp_out, out;

  if (world.rank() == 0) {
    in.resize(kCount);
    exp_out.resize(kCount);
    out.resize(kCount);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (size_t i = 0; i < kCount; i++) {
      in[i] = dist(gen);
      exp_out[i] = in[i];
    }
    std::ranges::sort(exp_out);
  }

  size_t local_size = 0;
  if (world.rank() == 0) {
    local_size = kCount / world.size();
  }
  boost::mpi::broadcast(world, local_size, 0);

  std::vector<double> local_in(local_size);
  if (world.rank() == 0) {
    std::vector<std::vector<double>> chunks;
    for (int i = 0; i < world.size(); ++i) {
      size_t start = i * local_size;
      size_t end = (i == world.size() - 1) ? kCount : start + local_size;
      chunks.emplace_back(in.begin() + start, in.begin() + end);
    }
    boost::mpi::scatter(world, chunks, local_in, 0);
  } else {
    boost::mpi::scatter(world, local_in, 0);
  }
  out.resize(local_size);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_in.data()));
  task_data->inputs_count.emplace_back(local_in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

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
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> gathered_out;
  if (world.rank() == 0) {
    gathered_out.resize(kCount);
    std::vector<std::vector<double>> chunks_to_gather(world.size());
    boost::mpi::gather(world, out, chunks_to_gather, 0);

    size_t current_pos = 0;
    for (const auto& chunk : chunks_to_gather) {
      std::copy(chunk.begin(), chunk.end(), gathered_out.begin() + current_pos);
      current_pos += chunk.size();
    }
  } else {
    boost::mpi::gather(world, out, 0);
  }
  for (size_t i = 0; i < gathered_out.size(); ++i) {
    std::cout << "Expected: " << exp_out[i] << ", Got: " << gathered_out[i] << "\n";
  }
  if (world.rank() == 0) {
    ASSERT_TRUE(std::equal(exp_out.begin(), exp_out.end(), gathered_out.begin(),
                           [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kCount = 10;
  std::vector<double> in, exp_out, out;

  if (world.rank() == 0) {
    in.resize(kCount);
    exp_out.resize(kCount);
    out.resize(kCount);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (size_t i = 0; i < kCount; i++) {
      in[i] = dist(gen);
      exp_out[i] = in[i];
    }
    std::ranges::sort(exp_out);
  }

  size_t local_size = 0;
  if (world.rank() == 0) {
    local_size = kCount / world.size();
  }
  boost::mpi::broadcast(world, local_size, 0);

  std::vector<double> local_in(local_size);
  if (world.rank() == 0) {
    std::vector<std::vector<double>> chunks;
    for (int i = 0; i < world.size(); ++i) {
      size_t start = i * local_size;
      size_t end = (i == world.size() - 1) ? kCount : start + local_size;
      chunks.emplace_back(in.begin() + start, in.begin() + end);
    }
    boost::mpi::scatter(world, chunks, local_in, 0);
  } else {
    boost::mpi::scatter(world, local_in, 0);
  }
  out.resize(local_size);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_in.data()));
  task_data->inputs_count.emplace_back(local_in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

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
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> gathered_out;
  if (world.rank() == 0) {
    gathered_out.resize(kCount);
    std::vector<std::vector<double>> chunks_to_gather(world.size());
    boost::mpi::gather(world, out, chunks_to_gather, 0);

    size_t current_pos = 0;
    for (const auto& chunk : chunks_to_gather) {
      std::copy(chunk.begin(), chunk.end(), gathered_out.begin() + current_pos);
      current_pos += chunk.size();
    }
  } else {
    boost::mpi::gather(world, out, 0);
  }
  for (size_t i = 0; i < gathered_out.size(); ++i) {
    std::cout << "Expected: " << exp_out[i] << ", Got: " << gathered_out[i] << "\n";
  }
  if (world.rank() == 0) {
    ASSERT_TRUE(std::equal(exp_out.begin(), exp_out.end(), gathered_out.begin(),
                           [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}