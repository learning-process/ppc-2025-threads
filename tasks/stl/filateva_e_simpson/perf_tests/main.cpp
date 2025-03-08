#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/filateva_e_simpson/include/ops_stl.hpp"

TEST(filateva_e_simpson_stl, test_pipeline_run) {
  std::vector<double> param = {1, 1000, 0.00005};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  auto test_task_sequential = std::make_shared<filateva_e_simpson_stl::Simpson>(task_data);

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

  filateva_e_simpson_stl::Func integral_f = [](double x) { return x * x * x / 3; };

  //std::cerr << "\n" << integral_f(param[1]) - integral_f(param[0]) - res[0] << "\n";

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_task_run) {
  std::vector<double> param = {1, 1000, 0.00005};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  auto test_task_sequential = std::make_shared<filateva_e_simpson_stl::Simpson>(task_data);

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

  filateva_e_simpson_stl::Func integral_f = [](double x) { return x * x * x / 3; };
  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}
