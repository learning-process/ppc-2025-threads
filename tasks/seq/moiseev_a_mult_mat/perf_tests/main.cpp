#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/moiseev_a_mult_mat/include/ops_seq.hpp"

namespace {

std::vector<double> GenerateRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (auto &val : matrix) {
    val = dist(gen);
  }
  return matrix;
}

}  // namespace

TEST(moiseev_a_mult_mat_seq, test_pipeline_run) {
  constexpr int kCount = 500;

  auto matrix_A = GenerateRandomMatrix(kCount, kCount);
  auto matrix_B = GenerateRandomMatrix(kCount, kCount);
  std::vector<double> matrix_C(kCount * kCount, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  task_data_seq->inputs_count.emplace_back(matrix_A.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  task_data_seq->inputs_count.emplace_back(matrix_B.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_C.data()));
  task_data_seq->outputs_count.emplace_back(matrix_C.size());

  auto test_task_sequential = std::make_shared<moiseev_a_mult_mat_seq::MultMatSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(moiseev_a_mult_mat_seq, test_task_run) {
  constexpr int kCount = 500;

  auto matrix_A = GenerateRandomMatrix(kCount, kCount);
  auto matrix_B = GenerateRandomMatrix(kCount, kCount);
  std::vector<double> matrix_C(kCount * kCount, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
  task_data_seq->inputs_count.emplace_back(matrix_A.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
  task_data_seq->inputs_count.emplace_back(matrix_B.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_C.data()));
  task_data_seq->outputs_count.emplace_back(matrix_C.size());

  auto test_task_sequential = std::make_shared<moiseev_a_mult_mat_seq::MultMatSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
