#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/shuravina_o_hoare_simple_merger_tbb/include/ops_tbb.hpp"

namespace {
std::vector<int> GenerateRandomVector(int size, int min_val, int max_val) {
  std::vector<int> data(size);
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> distrib(min_val, max_val);

  for (int i = 0; i < size; ++i) {
    data[i] = distrib(gen);
  }
  return data;
}
}  // namespace

TEST(shuravina_o_hoare_simple_merger_tbb, test_pipeline_run) {
  constexpr int kCount = 50000;
  constexpr int kMinValue = -100000;
  constexpr int kMaxValue = 100000;

  std::vector<int> in = GenerateRandomVector(kCount, kMinValue, kMaxValue);
  std::vector<int> out(kCount, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_tbb::TestTaskTBB>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    static auto start_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(shuravina_o_hoare_simple_merger_tbb, test_task_run) {
  constexpr int kCount = 50000;
  constexpr int kMinValue = -100000;
  constexpr int kMaxValue = 100000;

  std::vector<int> in = GenerateRandomVector(kCount, kMinValue, kMaxValue);
  std::vector<int> out(kCount, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    FAIL() << "Invalid task data";
  }

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_tbb::TestTaskTBB>(task_data);

  ASSERT_TRUE(test_task->Validation());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  perf_attr->current_timer = [&] {
    static auto start_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}