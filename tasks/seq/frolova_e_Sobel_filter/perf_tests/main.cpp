#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/frolova_e_Sobel_filter/include/ops_seq.hpp"

TEST(frolova_e_Sobel_filter_seq, test_pipeline_run) {
  std::vector<int> value_1 = {2000, 2000};
  std::vector<int> pict = frolova_e_Sobel_filter_seq::genRGBpicture(2000, 2000, 0);

  std::vector<int> res(4000000, 0);

  std::vector<int> reference(4000000, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
  task_data_seq->inputs_count.emplace_back(value_1.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_seq->inputs_count.emplace_back(pict.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  auto test_task_sequential = std::make_shared<frolova_e_Sobel_filter_seq::SobelFilterSequential>(task_data_seq);

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
  ASSERT_EQ(4000000, res.size());
}

TEST(frolova_e_Sobel_filter_seq, test_task_run) {
  std::vector<int> value_1 = {2000, 2000};
  std::vector<int> pict = frolova_e_Sobel_filter_seq::genRGBpicture(2000, 2000, 0);

  std::vector<int> res(4000000, 0);

  std::vector<int> reference(4000000, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
  task_data_seq->inputs_count.emplace_back(value_1.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_seq->inputs_count.emplace_back(pict.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  auto test_task_sequential = std::make_shared<frolova_e_Sobel_filter_seq::SobelFilterSequential>(task_data_seq);

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
}