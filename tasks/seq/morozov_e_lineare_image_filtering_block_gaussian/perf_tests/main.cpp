#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/morozov_e_lineare_image_filtering_block_gaussian/include/ops_seq.hpp"

TEST(morozov_e_lineare_image_filtering_block_gaussian_seq, test_pipeline_run) {
  int n = 4000;
  int m = 4000;
  std::vector<std::vector<double>> image(n, std::vector<double>(m, 1));
  std::vector image_res(n, std::vector<double>(m, 1));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; j++) {
      image_res[i][j] = 1;
    }
  }
  std::vector real_res(n, std::vector<double>(m, 1));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  auto test_task_sequential =
      std::make_shared<morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(image_res, real_res);
}

TEST(morozov_e_lineare_image_filtering_block_gaussian_seq, test_task_run) {
  int n = 4000;
  int m = 4000;
  std::vector<std::vector<double>> image(n, std::vector<double>(m, 1));
  std::vector image_res(n, std::vector<double>(m, 1));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; j++) {
      image_res[i][j] = 1;
    }
  }
  std::vector real_res(n, std::vector<double>(m, 1));

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image[i].data()));
  }
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(m);
  for (int i = 0; i < n; ++i) {
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(image_res[i].data()));
  }
  task_data_seq->outputs_count.emplace_back(n);
  task_data_seq->outputs_count.emplace_back(m);

  // Create Task
  auto test_task_sequential =
      std::make_shared<morozov_e_lineare_image_filtering_block_gaussian::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(image_res, real_res);
}
