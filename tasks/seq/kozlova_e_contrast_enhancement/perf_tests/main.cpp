#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kozlova_e_contrast_enhancement/include/ops_seq.hpp"

std::vector<int> generate_vector(int length);

std::vector<int> generate_vector(int length) {
  std::vector<int> vec;
  for (int i = 0; i < length; ++i) {
    vec.push_back(rand() % 256);
  }
  return vec;
}

TEST(kozlova_e_contrast_enhancement_seq, test_pipeline_run) {
  constexpr int size = 20000000;

  // Create data
  std::vector<int> in = generate_vector(size);
  std::vector<int> out(size, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kozlova_e_contrast_enhancement_seq::TestTaskSequential>(task_data_seq);

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

  int min_value = *std::min_element(in.begin(), in.end());
  int max_value = *std::max_element(in.begin(), in.end());

  for (size_t i = 0; i < in.size(); ++i) {
    int expected = static_cast<int>(((in[i] - min_value) / (double)(max_value - min_value)) * 255);
    expected = std::clamp(expected, 0, 255);
    EXPECT_EQ(out[i], expected);
  }
}

TEST(kozlova_e_contrast_enhancement_seq, test_task_run) {
  constexpr int size = 20000000;

  // Create data
  std::vector<int> in = generate_vector(size);
  std::vector<int> out(size, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kozlova_e_contrast_enhancement_seq::TestTaskSequential>(task_data_seq);

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

  int min_value = *std::min_element(in.begin(), in.end());
  int max_value = *std::max_element(in.begin(), in.end());

  for (size_t i = 0; i < in.size(); ++i) {
    int expected = static_cast<int>(((in[i] - min_value) / (double)(max_value - min_value)) * 255);
    expected = std::clamp(expected, 0, 255);
    EXPECT_EQ(out[i], expected);
  }
}
