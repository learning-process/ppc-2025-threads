#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/gromov_a_fox_algorithm/include/ops_stl.hpp"

namespace {
std::vector<double> NaiveMatrixMultiply(const std::vector<double>& a, const std::vector<double>& b, size_t n) {
  if (n == 0) {
    return {};
  }
  std::vector<double> result(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        result[(i * n) + j] += a[(i * n) + k] * b[(k * n) + j];
      }
    }
  }
  return result;
}
}  // namespace

TEST(gromov_a_fox_algorithm_stl, test_pipeline_run) {
  constexpr size_t kN = 400;

  std::vector<double> a(kN * kN);
  std::vector<double> b(kN * kN);
  std::vector<double> out(kN * kN, 0.0);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      a[(i * kN) + j] = static_cast<double>(i + j + 1);
      b[(i * kN) + j] = static_cast<double>(kN - i + j + 1);
    }
  }

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<gromov_a_fox_algorithm_stl::TestTaskSTL>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> expected = NaiveMatrixMultiply(a, b, kN);

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-3);
  }
}

TEST(gromov_a_fox_algorithm_stl, test_task_run) {
  constexpr size_t kN = 400;

  std::vector<double> a(kN * kN);
  std::vector<double> b(kN * kN);
  std::vector<double> out(kN * kN, 0.0);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      a[(i * kN) + j] = static_cast<double>(i + j + 1);
      b[(i * kN) + j] = static_cast<double>(kN - i + j + 1);
    }
  }

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<gromov_a_fox_algorithm_stl::TestTaskSTL>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  std::vector<double> expected = NaiveMatrixMultiply(a, b, kN);

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-3);
  }
}