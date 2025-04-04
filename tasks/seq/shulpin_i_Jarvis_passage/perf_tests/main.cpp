#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shulpin_i_Jarvis_passage/include/ops_seq.hpp"

namespace {
std::vector<shulpin_i_jarvis_seq::Point> GenerateRandomPoints(size_t num_points) {
  std::vector<shulpin_i_jarvis_seq::Point> points;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000, 1000);

  for (size_t i = 0; i < num_points; ++i) {
    double x = dist(gen);
    double y = dist(gen);
    points.emplace_back(shulpin_i_jarvis_seq::Point{x, y});
  }

  return points;
}

int Orientation(const shulpin_i_jarvis_seq::Point& p, const shulpin_i_jarvis_seq::Point& q,
                const shulpin_i_jarvis_seq::Point& r) {
  double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
  if (std::fabs(val) < 1e-9) return 0;
  return (val > 0) ? 1 : 2;
}

std::vector<shulpin_i_jarvis_seq::Point> ComputeConvexHull(std::vector<shulpin_i_jarvis_seq::Point> raw_points) {
  std::vector<shulpin_i_jarvis_seq::Point> convex_shell{};
  const size_t count = raw_points.size();

  size_t ref_idx = 0;
  for (size_t idx = 1; idx < count; ++idx) {
    const auto& p = raw_points[idx];
    const auto& ref = raw_points[ref_idx];
    if ((p.x < ref.x) || (p.x == ref.x && p.y < ref.y)) {
      ref_idx = idx;
    }
  }

  std::vector<bool> included(count, false);
  size_t current = ref_idx;

  while (true) {
    convex_shell.push_back(raw_points[current]);
    included[current] = true;

    size_t next = (current + 1) % count;

    for (size_t trial = 0; trial < count; ++trial) {
      if (trial == current || trial == next) continue;

      int orient = Orientation(raw_points[current], raw_points[trial], raw_points[next]);
      if (orient == 2) {
        next = trial;
      }
    }

    current = next;
    if (current == ref_idx) break;
  }
  return convex_shell;
}
}  // namespace

TEST(shulpin_i_jarvis_seq, test_pipeline_run) {
  size_t num_points = 10000;
  std::vector<shulpin_i_jarvis_seq::Point> input = GenerateRandomPoints(num_points);
  
  std::vector<shulpin_i_jarvis_seq::Point> out{};
  out.reserve(input.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<shulpin_i_jarvis_seq::JarvisSequential>(task_data_seq);

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
  
  std::vector<shulpin_i_jarvis_seq::Point> expected = ComputeConvexHull(input);

  for (const auto& p : out) {
    bool found = false;
    for (const auto& q : expected) {
      if (std::fabs(p.x - q.x) < 1e-6 && std::fabs(p.y - q.y) < 1e-6) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}

TEST(shulpin_i_jarvis_seq, test_task_run) {
  size_t num_points = 10000;

  std::vector<shulpin_i_jarvis_seq::Point> input = GenerateRandomPoints(num_points);

  std::vector<shulpin_i_jarvis_seq::Point> out{};
  out.reserve(input.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<shulpin_i_jarvis_seq::JarvisSequential>(task_data_seq);

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

  std::vector<shulpin_i_jarvis_seq::Point> expected = ComputeConvexHull(input);

  for (const auto& p : out) {
    bool found = false;
    for (const auto& q : expected) {
      if (std::fabs(p.x - q.x) < 1e-6 && std::fabs(p.y - q.y) < 1e-6) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}
