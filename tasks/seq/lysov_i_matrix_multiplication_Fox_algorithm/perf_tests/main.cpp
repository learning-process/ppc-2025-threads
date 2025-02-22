#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"

std::vector<double> getRandomMatrix(size_t size) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(1e-3, 1e3);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_pipeline_run) {
  // Create data
  size_t N = 512;
  size_t block_size = 100;
  std::vector<double> A = getRandomMatrix(N);
  std::vector<double> B = getRandomMatrix(N);
  std::vector<double> C(N * N, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);

  // Create Task
  auto test_task_sequential =
      std::make_shared<lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential>(taskDataSeq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_task_run) {
  size_t N = 512;
  size_t block_size = 100;
  std::vector<double> A = getRandomMatrix(N);
  std::vector<double> B = getRandomMatrix(N);
  std::vector<double> C(N * N, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);

  // Create Task
  auto test_task_sequential =
      std::make_shared<lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential>(taskDataSeq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
