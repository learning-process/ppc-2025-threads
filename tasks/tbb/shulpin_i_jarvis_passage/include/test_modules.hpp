#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/shulpin_i_jarvis_passage/include/ops_tbb.hpp"

namespace {
void VerifyResults(const std::vector<shulpin_i_jarvis_tbb::Point> &expected,
                   const std::vector<shulpin_i_jarvis_tbb::Point> &result_seq,
                   const std::vector<shulpin_i_jarvis_tbb::Point> &result_tbb) {
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i].x, result_seq[i].x);
    ASSERT_EQ(expected[i].y, result_seq[i].y);
    ASSERT_EQ(expected[i].x, result_tbb[i].x);
    ASSERT_EQ(expected[i].y, result_tbb[i].y);
  }
}

[[maybe_unused]] void MainTestBody(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                   std::vector<shulpin_i_jarvis_tbb::Point> &expected) {
  std::vector<shulpin_i_jarvis_tbb::Point> result_seq(expected.size());
  std::vector<shulpin_i_jarvis_tbb::Point> result_tbb(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result_seq.size()));

  shulpin_i_jarvis_tbb::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_tbb::JarvisTBBParallel tbb_task(task_data_par);
  ASSERT_EQ(tbb_task.Validation(), true);
  tbb_task.PreProcessing();
  tbb_task.Run();
  tbb_task.PostProcessing();

  VerifyResults(expected, result_seq, result_tbb);
}

inline size_t CalculateIndex(size_t i, size_t tmp) { return (i < tmp) ? (i + tmp) : (i - tmp); }

inline void ExpectEqualPoints(const shulpin_i_jarvis_tbb::Point &expected, const shulpin_i_jarvis_tbb::Point &seq,
                              const shulpin_i_jarvis_tbb::Point &tbb) {
  EXPECT_EQ(expected.x, seq.x);
  EXPECT_EQ(expected.y, seq.y);
  EXPECT_EQ(expected.x, tbb.x);
  EXPECT_EQ(expected.y, tbb.y);
}

void VerifyResultsCircle(const std::vector<shulpin_i_jarvis_tbb::Point> &expected,
                         const std::vector<shulpin_i_jarvis_tbb::Point> &result_seq,
                         const std::vector<shulpin_i_jarvis_tbb::Point> &result_tbb, size_t &num_points) {
  size_t tmp = num_points >> 1;

  for (size_t i = 0; i < expected.size(); ++i) {
    size_t idx = CalculateIndex(i, tmp);
    ExpectEqualPoints(expected[i], result_seq[idx], result_tbb[idx]);
  }
}

[[maybe_unused]] void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                           std::vector<shulpin_i_jarvis_tbb::Point> &expected, size_t &num_points) {
  std::vector<shulpin_i_jarvis_tbb::Point> result_seq(expected.size());
  std::vector<shulpin_i_jarvis_tbb::Point> result_tbb(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result_seq.size()));

  shulpin_i_jarvis_tbb::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_tbb::JarvisTBBParallel tbb_task(task_data_par);
  ASSERT_EQ(tbb_task.Validation(), true);
  tbb_task.PreProcessing();
  tbb_task.Run();
  tbb_task.PostProcessing();

  VerifyResultsCircle(expected, result_seq, result_tbb, num_points);
}

[[maybe_unused]] void TestBodyFalse(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                    std::vector<shulpin_i_jarvis_tbb::Point> &expected) {
  std::vector<shulpin_i_jarvis_tbb::Point> result_tbb(expected.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_tbb::JarvisTBBParallel tbb_task(task_data_par);
  ASSERT_EQ(tbb_task.Validation(), false);
}

int Orientation(const shulpin_i_jarvis_tbb::Point &p, const shulpin_i_jarvis_tbb::Point &q,
                const shulpin_i_jarvis_tbb::Point &r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

[[maybe_unused]] std::vector<shulpin_i_jarvis_tbb::Point> ComputeConvexHull(
    std::vector<shulpin_i_jarvis_tbb::Point> raw_points) {
  std::vector<shulpin_i_jarvis_tbb::Point> convex_shell{};
  const size_t count = raw_points.size();

  size_t ref_idx = 0;
  for (size_t idx = 1; idx < count; ++idx) {
    const auto &p = raw_points[idx];
    const auto &ref = raw_points[ref_idx];
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
      if (trial == current || trial == next) {
        continue;
      }

      int orient = Orientation(raw_points[current], raw_points[trial], raw_points[next]);
      if (orient == 2) {
        next = trial;
      }
    }

    current = next;
    if (current == ref_idx) {
      break;
    }
  }
  return convex_shell;
}

void VerifyResultsRandom(const std::vector<shulpin_i_jarvis_tbb::Point> &expected,
                         const std::vector<shulpin_i_jarvis_tbb::Point> &result_tbb) {
  for (const auto &p : result_tbb) {
    bool found = false;
    for (const auto &q : expected) {
      if (std::fabs(p.x - q.x) < 1e-6 && std::fabs(p.y - q.y) < 1e-6) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}

[[maybe_unused]] void RandomTestBody(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                     std::vector<shulpin_i_jarvis_tbb::Point> &expected) {
  std::vector<shulpin_i_jarvis_tbb::Point> result_tbb(expected.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_tbb::JarvisTBBParallel tbb_task(task_data_par);
  ASSERT_EQ(tbb_task.Validation(), true);
  tbb_task.PreProcessing();
  tbb_task.Run();
  tbb_task.PostProcessing();

  VerifyResultsRandom(expected, result_tbb);
}
}  // namespace