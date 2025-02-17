#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shulpin_i_Jarvis_passage/include/ops_seq.hpp"

namespace shulpin_i_Jarvis_seq {
std::vector<shulpin_i_Jarvis_seq::Point> generatePointsInCircle(const shulpin_i_Jarvis_seq::Point &center,
                                                                double radius, size_t numPoints) {
  std::vector<shulpin_i_Jarvis_seq::Point> points;

  for (size_t i = 0; i < numPoints; ++i) {
    double angle = 2.0 * M_PI * i / numPoints;
    double x = center.x + radius * cos(angle);
    double y = center.y + radius * sin(angle);
    points.push_back({x,y});
  }

  return points;
}
}  // namespace shulpin_i_Jarvis_seq

TEST(shulpin_i_Jarvis_seq, test_pipeline_run) {
  shulpin_i_Jarvis_seq::Point center{0, 0};
  double radius = 10.0;
  size_t numPoints = 10000;

  std::vector<shulpin_i_Jarvis_seq::Point> input =
      shulpin_i_Jarvis_seq::generatePointsInCircle(center, radius, numPoints);

  std::vector<shulpin_i_Jarvis_seq::Point> out(input.size());
  std::vector<shulpin_i_Jarvis_seq::Point> expected = input;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<shulpin_i_Jarvis_seq::JarvisSequential>(task_data_seq);

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

  size_t tmp = numPoints >> 1;

  for (size_t i = 0; i < out.size(); ++i) {
    size_t idx = (i < tmp) ? (i + tmp) : (i - tmp);
    EXPECT_EQ(expected[i].x, out[idx].x);
    EXPECT_EQ(expected[i].y, out[idx].y);
  }
}

TEST(shulpin_i_Jarvis_seq, test_task_run) {
  shulpin_i_Jarvis_seq::Point center{0, 0};
  double radius = 10.0;
  size_t numPoints = 10000;

  std::vector<shulpin_i_Jarvis_seq::Point> input =
      shulpin_i_Jarvis_seq::generatePointsInCircle(center, radius, numPoints);

  std::vector<shulpin_i_Jarvis_seq::Point> out(input.size());
  std::vector<shulpin_i_Jarvis_seq::Point> expected = input;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<shulpin_i_Jarvis_seq::JarvisSequential>(task_data_seq);

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

  size_t tmp = numPoints >> 1;

  for (size_t i = 0; i < out.size(); ++i) {
    size_t idx = (i < tmp) ? (i + tmp) : (i - tmp);
    EXPECT_EQ(expected[i].x, out[idx].x);
    EXPECT_EQ(expected[i].y, out[idx].y);
  }
}