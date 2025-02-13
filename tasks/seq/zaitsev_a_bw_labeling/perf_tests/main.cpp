#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

namespace {
std::vector<int> GenInVector(int size) {
  std::vector<int> in(size, 0);
  for (int i = 0; i < size; i += 2) {
    in[i] = 1;
  }
  return in;
}

std::vector<int> GenOutVector(int size) {
  std::vector<int> out(size, 0);
  for (int i = 0; i < size; i += 2) {
    out[i] = (i + 2) / 2;
  }
  return out;
}
}  // namespace

TEST(zaitsev_a_labeling_seq, test_pipeline_run) {
  constexpr int kW = 9999;
  constexpr int kH = 9999;

  constexpr int kSize = kW * kH;

  // Create data
  std::vector<int> in = ::GenInVector(kSize);
  std::vector<int> out = std::vector<int>(kSize);
  std::vector<int> expected = ::GenOutVector(kSize);

  // Create task_data

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(in.data())));
  task_data_seq->inputs_count.emplace_back(kW);
  task_data_seq->inputs_count.emplace_back(kH);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  auto test_task_sequential = std::make_shared<zaitsev_a_labeling::Labeler>(task_data_seq);

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
  ASSERT_EQ(expected, out);
  EXPECT_EQ(1, 1);
}

TEST(zaitsev_a_labeling_seq, test_task_run) {
  constexpr int kW = 9999;
  constexpr int kH = 9999;

  constexpr int kSize = kW * kH;

  // Create data
  std::vector<int> in = ::GenInVector(kSize);
  std::vector<int> out = std::vector<int>(kSize);
  std::vector<int> expected = ::GenOutVector(kSize);

  // Create task_data

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(in.data())));
  task_data_seq->inputs_count.emplace_back(kW);
  task_data_seq->inputs_count.emplace_back(kH);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  auto test_task_sequential = std::make_shared<zaitsev_a_labeling::Labeler>(task_data_seq);

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
  EXPECT_EQ(expected, out);
}
