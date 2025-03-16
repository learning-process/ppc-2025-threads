
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq/include/ops_seq.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq, test_pipeline_run) {
  constexpr int kCount = 50000;

  std::vector<int> input(kCount, 0);
  std::vector<int> output(kCount, 0);

  std::srand(42);
  for (size_t i = 0; i < kCount; ++i) {
    input[i] = std::rand() % 1000;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  auto test_task_sequential =
      std::make_shared<fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<int> expected_output = input;
  std::sort(expected_output.begin(), expected_output.end());
  ASSERT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq, test_task_run) {
  constexpr int kCount = 50000;

  std::vector<int> input(kCount, 0);
  std::vector<int> output(kCount, 0);

  std::srand(42);
  for (size_t i = 0; i < kCount; ++i) {
    input[i] = std::rand() % 1000;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  auto test_task_sequential =
      std::make_shared<fyodorov_m_shell_sort_with_even_odd_batcher_merge_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<int> expected_output = input;
  std::sort(expected_output.begin(), expected_output.end());
  ASSERT_EQ(output, expected_output);
}
