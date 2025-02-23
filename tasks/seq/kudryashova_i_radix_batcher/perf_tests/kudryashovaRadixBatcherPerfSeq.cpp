#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherSeq.hpp"

std::vector<double> GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(kudryashova_i_radix_batcher_seq, test_pipeline_run) {
  int global_vector_size = 3000000;
  std::vector<double> global_vector = GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  auto testTaskSequential = std::make_shared<kudryashova_i_radix_batcher_seq::TestTaskSequential>(taskData);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(kudryashova_i_radix_batcher_seq, test_task_run) {
  int global_vector_size = 3000000;
  std::vector<double> global_vector = GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  auto testTaskSequential = std::make_shared<kudryashova_i_radix_batcher_seq::TestTaskSequential>(taskData);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}
