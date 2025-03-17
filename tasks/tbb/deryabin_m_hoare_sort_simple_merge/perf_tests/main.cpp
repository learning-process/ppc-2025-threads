#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/deryabin_m_hoare_sort_simple_merge/include/ops_omp.hpp"

TEST(deryabin_m_hoare_sort_simple_merge_omp, test_pipeline_run_Omp) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(512000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 256;
  std::vector<double> output_array(512000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::vector<std::vector<double>> true_sol(1, true_solution);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_omp->inputs_count.emplace_back(input_array.size());
  task_data_omp->inputs_count.emplace_back(chunk_count);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_omp->outputs_count.emplace_back(output_array.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(true_sol.data()));
  task_data_seq->outputs_count.emplace_back(true_solution.size());
  deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();

  auto hoare_sort_simple_merge_task_omp =
      std::make_shared<deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskOpenMP>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true_sol[0], out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_omp, test_task_run_Omp) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(512000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 256;
  std::vector<double> output_array(512000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::vector<std::vector<double>> true_sol(1, true_solution);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_omp->inputs_count.emplace_back(input_array.size());
  task_data_omp->inputs_count.emplace_back(chunk_count);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_omp->outputs_count.emplace_back(output_array.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(true_sol.data()));
  task_data_seq->outputs_count.emplace_back(true_solution.size());
  deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();

  auto hoare_sort_simple_merge_task_omp =
      std::make_shared<deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskOpenMP>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true_sol[0], out_array[0]);
}
