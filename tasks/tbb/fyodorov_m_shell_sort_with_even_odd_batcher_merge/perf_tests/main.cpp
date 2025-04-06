#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_tbb.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_pipeline_run) {
  constexpr int kCount = 520000;

  auto get_random_vector = [](int sz, int a, int b) -> std::vector<int> {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> dis(a, b);
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) {
      vec[i] = dis(gen);
    }
    return vec;
  };

  std::vector<int> input = get_random_vector(kCount, 0, 999);
  std::vector<int> output(kCount, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  auto test_task_tbb =
      std::make_shared<fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<int> expected_output = input;
  std::ranges::sort(expected_output);
  ASSERT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb, test_task_run) {
  constexpr int kCount = 520000;

  auto get_random_vector = [](int sz, int a, int b) -> std::vector<int> {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> dis(a, b);
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) {
      vec[i] = dis(gen);
    }
    return vec;
  };

  std::vector<int> input = get_random_vector(kCount, 0, 999);
  std::vector<int> output(kCount, 0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_tbb->inputs_count.emplace_back(input.size());
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_tbb->outputs_count.emplace_back(output.size());

  auto test_task_tbb =
      std::make_shared<fyodorov_m_shell_sort_with_even_odd_batcher_merge_tbb::TestTaskTBB>(task_data_tbb);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() -> double {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_tbb);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<int> expected_output = input;
  std::ranges::sort(expected_output);
  ASSERT_EQ(output, expected_output);
}