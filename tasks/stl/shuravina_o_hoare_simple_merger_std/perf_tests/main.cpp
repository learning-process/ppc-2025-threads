#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp>
#include <thread>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(shuravina_o_hoare_simple_merger_stl, test_pipeline_run) {
  constexpr int count = 50000;
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);

  for (int i = 0; i < count; i++) {
    in[i] = count - i;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_stl::TestTaskSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  auto start = std::chrono::high_resolution_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, nullptr);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  if (duration < 0.05) {
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>((0.05 - duration) * 1000)));
  }

  for (int i = 0; i < count - 1; i++) {
    ASSERT_LE(out[i], out[i + 1]);
  }
}

TEST(shuravina_o_hoare_simple_merger_stl, test_task_run) {
  constexpr int count = 50000;
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);

  for (int i = 0; i < count; i++) {
    in[i] = count - i;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<shuravina_o_hoare_simple_merger_stl::TestTaskSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  auto start = std::chrono::high_resolution_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, nullptr);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  if (duration < 0.05) {
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>((0.05 - duration) * 1000)));
  }

  for (int i = 0; i < count - 1; i++) {
    ASSERT_LE(out[i], out[i + 1]);
  }
}