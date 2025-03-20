#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/Lavrentiev_A_CCS_SEQ/include/ops_seq.hpp"

namespace {
constexpr auto kEpsilon = 0.000001;
struct TestData {
  std::vector<double> random_data;
  std::vector<double> single_matrix;
  std::vector<double> result;
  std::shared_ptr<ppc::core::TaskData> task_data_seq;
  TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size);
};

TestData::TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size) {
  random_data = std::move(lavrentiev_a_ccs_seq::GenerateRandomMatrix(matrix1_size.first * matrix1_size.second));
  single_matrix = std::move(lavrentiev_a_ccs_seq::GenerateSingleMatrix(matrix2_size.first * matrix2_size.second));
  result.resize(matrix1_size.first * matrix2_size.second);
  task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(random_data.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix1_size.first);
  task_data_seq->inputs_count.emplace_back(matrix1_size.second);
  task_data_seq->inputs_count.emplace_back(matrix2_size.first);
  task_data_seq->inputs_count.emplace_back(matrix2_size.second);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data_seq->outputs_count.emplace_back(result.size());
}

}  // namespace

TEST(lavrentiev_a_ccs_seq, test_pipeline_run) {
  constexpr double kSize = 500;
  auto task = TestData({kSize, kSize}, {kSize, kSize});
  auto test_task_sequential = std::make_shared<lavrentiev_a_ccs_seq::CCSSequential>(task.task_data_seq);
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
  for (size_t i = 0; i < task.result.size(); ++i) {
    EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
  }
}

TEST(lavrentiev_a_ccs_seq, test_task_run) {
  constexpr double kSize = 500;
  auto task = TestData({kSize, kSize}, {kSize, kSize});
  auto test_task_sequential = std::make_shared<lavrentiev_a_ccs_seq::CCSSequential>(task.task_data_seq);
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
  for (size_t i = 0; i < task.result.size(); ++i) {
    EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
  }
}
