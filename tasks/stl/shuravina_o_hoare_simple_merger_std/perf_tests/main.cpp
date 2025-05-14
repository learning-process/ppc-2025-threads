#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(shuravina_o_hoare_simple_merger_stl, test_pipeline_run) {
  int count = 50000;
  double perf_time = 0.0;
  std::shared_ptr<ppc::core::PerfResults> perf_results;

  do {
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
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point.time_since_epoch()).count();
      return static_cast<double>(duration) * 1e-9;
    };

    perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

    auto start = std::chrono::high_resolution_clock::now();
    perf_analyzer->PipelineRun(perf_attr, perf_results);
    auto end = std::chrono::high_resolution_clock::now();

    perf_time = std::chrono::duration<double>(end - start).count();
    count += 5000;

  } while (perf_time < 0.05);

  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(shuravina_o_hoare_simple_merger_stl, test_task_run) {
  int count = 50000;
  double perf_time = 0.0;
  std::shared_ptr<ppc::core::PerfResults> perf_results;

  do {
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
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point.time_since_epoch()).count();
      return static_cast<double>(duration) * 1e-9;
    };

    perf_results = std::make_shared<ppc::core::PerfResults>();

    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

    auto start = std::chrono::high_resolution_clock::now();
    perf_analyzer->TaskRun(perf_attr, perf_results);
    auto end = std::chrono::high_resolution_clock::now();

    perf_time = std::chrono::duration<double>(end - start).count();
    count += 5000;

  } while (perf_time < 0.05);

  ppc::core::Perf::PrintPerfStatistic(perf_results);
}