#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/borisov_s_strassen_seq/include/ops_seq.hpp"

namespace {

void GenerateRandomMatrix(int rows, int cols, std::vector<double>& matrix) {
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  matrix.resize(rows * cols);
  for (auto& value : matrix) {
    value = dist(rng);
  }
}

}  // namespace

TEST(borisov_s_seq_strassen_perf, test_pipeline_run) {
  constexpr int rowsA = 1024;
  constexpr int colsA = 512;
  constexpr int rowsB = 512;
  constexpr int colsB = 1024;

  std::vector<double> A;
  std::vector<double> B;
  GenerateRandomMatrix(rowsA, colsA, A);
  GenerateRandomMatrix(rowsB, colsB, B);

  std::vector<double> in_data = {static_cast<double>(rowsA), static_cast<double>(colsA), static_cast<double>(rowsB),
                                 static_cast<double>(colsB)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (rowsA * colsB);
  std::vector<double> out(output_count, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_seq->inputs_count.emplace_back(in_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<borisov_s_strassen_seq::SequentialStrassenSeq>(task_data_seq);

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

TEST(borisov_s_seq_strassen_perf, test_task_run) {
  constexpr int rowsA = 1024;
  constexpr int colsA = 512;
  constexpr int rowsB = 512;
  constexpr int colsB = 1024;

  std::vector<double> A;
  std::vector<double> B;
  GenerateRandomMatrix(rowsA, colsA, A);
  GenerateRandomMatrix(rowsB, colsB, B);

  std::vector<double> in_data = {static_cast<double>(rowsA), static_cast<double>(colsA), static_cast<double>(rowsB),
                                 static_cast<double>(colsB)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (rowsA * colsB);
  std::vector<double> out(output_count, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_seq->inputs_count.emplace_back(in_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<borisov_s_strassen_seq::SequentialStrassenSeq>(task_data_seq);

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
