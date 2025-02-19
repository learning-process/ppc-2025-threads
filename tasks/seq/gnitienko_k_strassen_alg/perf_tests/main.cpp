#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/gnitienko_k_strassen_alg/include/ops_seq.hpp"

namespace gnitienko_k_matrix_func {
double minVal = -50.0;
double maxVal = 50.0;
std::vector<double> genMatrix(size_t size);
void TrivialMultiply(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, int size);

std::vector<double> genMatrix(size_t size) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(minVal, maxVal);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] = dist(gen);
    }
  }
  return matrix;
}

void TrivialMultiply(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      C[i * size + j] = 0;
      for (size_t k = 0; k < size; ++k) {
        C[i * size + j] += A[i * size + k] * B[k * size + j];
        C[i * size + j] = round(C[i * size + j] * 10000) / 10000;
      }
    }
  }
}
}  // namespace gnitienko_k_matrix_func

TEST(gnitienko_k_strassen_alg_seq, test_pipeline_run) {
  size_t size = 512;

  // Create data
  std::vector<double> A = gnitienko_k_matrix_func::genMatrix(size);
  std::vector<double> B = gnitienko_k_matrix_func::genMatrix(size);
  std::vector<double> expected(size * size);
  gnitienko_k_matrix_func::TrivialMultiply(A, B, expected, size);
  std::vector<double> out(size * size);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<gnitienko_k_strassen_algorithm::StrassenAlgSeq>(task_data_seq);

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
  for (size_t i = 0; i < size * size; i++) EXPECT_NEAR(expected[i], out[i], 1e-2);
}

TEST(gnitienko_k_strassen_alg_seq, test_task_run) {
  size_t size = 512;
  // Create data
  std::vector<double> A = gnitienko_k_matrix_func::genMatrix(size);
  std::vector<double> B = gnitienko_k_matrix_func::genMatrix(size);
  std::vector<double> expected(size * size);
  gnitienko_k_matrix_func::TrivialMultiply(A, B, expected, size);
  std::vector<double> out(size * size);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<gnitienko_k_strassen_algorithm::StrassenAlgSeq>(task_data_seq);

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
  for (size_t i = 0; i < size * size; i++) EXPECT_NEAR(expected[i], out[i], 1e-2);
}
