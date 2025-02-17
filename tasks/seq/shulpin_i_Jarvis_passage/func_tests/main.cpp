#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/shulpin_i_Jarvis_passage/include/ops_seq.hpp"

namespace shulpin_i_Jarvis_seq {
std::vector<shulpin_i_Jarvis_seq::Point> generatePointsInCircle(const shulpin_i_Jarvis_seq::Point& center,
                                                                double radius, uint32_t numPoints) {
  std::vector<shulpin_i_Jarvis_seq::Point> points;

  for (uint32_t i = 0; i < numPoints; ++i) {
    double angle = 2.0 * M_PI * i / numPoints;
    double x = center.x + radius * cos(angle);
    double y = center.y + radius * sin(angle);
    points.push_back({x, y});
  }

  return points;
}

void TestBody(std::vector<shulpin_i_Jarvis_seq::Point>& input, std::vector<shulpin_i_Jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_Jarvis_seq::Point> result(expected.size());

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_Jarvis_seq::JarvisSequential SeqTask(taskDataSeq);
  ASSERT_EQ(SeqTask.Validation(), true);
  SeqTask.PreProcessing();
  SeqTask.Run();
  SeqTask.PostProcessing();

  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(expected[i].x, result[i].x);
    ASSERT_EQ(expected[i].y, result[i].y);
  }
}

void TestBody_False(std::vector<shulpin_i_Jarvis_seq::Point>& input,
                    std::vector<shulpin_i_Jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_Jarvis_seq::Point> result(expected.size());

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_Jarvis_seq::JarvisSequential SeqTask(taskDataSeq);
  ASSERT_EQ(SeqTask.Validation(), false);
}

void TestBody_Random_Circle(std::vector<shulpin_i_Jarvis_seq::Point>& input,
                            std::vector<shulpin_i_Jarvis_seq::Point>& expected, size_t& numPoints) {
  std::vector<shulpin_i_Jarvis_seq::Point> result(expected.size());

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_Jarvis_seq::JarvisSequential SeqTask(taskDataSeq);
  ASSERT_EQ(SeqTask.Validation(), true);
  SeqTask.PreProcessing();
  SeqTask.Run();
  SeqTask.PostProcessing();

  size_t tmp = numPoints >> 1;

  for (size_t i = 0; i < result.size(); ++i) {
    size_t idx = (i < tmp) ? (i + tmp) : (i - tmp);
    EXPECT_EQ(expected[i].x, result[idx].x);
    EXPECT_EQ(expected[i].y, result[idx].y);
  }
}
}  // namespace shulpin_i_Jarvis_seq

TEST(shulpin_i_Jarvis_seqq, square_with_point) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  shulpin_i_Jarvis_seq::TestBody(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, triangle) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{0, 0}, {3, 0}, {1, 2}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{0, 0}, {3, 0}, {1, 2}};

  shulpin_i_Jarvis_seq::TestBody(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, octagone) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};

  shulpin_i_Jarvis_seq::TestBody(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, repeated_points) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  shulpin_i_Jarvis_seq::TestBody(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, real_case) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};

  shulpin_i_Jarvis_seq::TestBody(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, one_point_validation_false) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{0, 0}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{0, 0}};

  shulpin_i_Jarvis_seq::TestBody_False(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, three_points_validation_false) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {{1, 1}, {2, 2}};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {{1, 1}, {2, 2}};

  shulpin_i_Jarvis_seq::TestBody_False(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, zero_points_validation_false) {
  std::vector<shulpin_i_Jarvis_seq::Point> input = {};
  std::vector<shulpin_i_Jarvis_seq::Point> expected = {};

  shulpin_i_Jarvis_seq::TestBody_False(input, expected);
}

TEST(shulpin_i_Jarvis_seqq, circle_r10_p100) {
  shulpin_i_Jarvis_seq::Point center{0, 0};
  double radius = 10.0;
  size_t numPoints = 100;
  std::vector<shulpin_i_Jarvis_seq::Point> input =
      shulpin_i_Jarvis_seq::generatePointsInCircle(center, radius, numPoints);
  std::vector<shulpin_i_Jarvis_seq::Point> expected = input;

  shulpin_i_Jarvis_seq::TestBody_Random_Circle(input, expected, numPoints);
}

TEST(shulpin_i_Jarvis_seqq, circle_r10_p1000) {
  shulpin_i_Jarvis_seq::Point center{0, 0};
  double radius = 10.0;
  size_t numPoints = 1000;
  std::vector<shulpin_i_Jarvis_seq::Point> input =
      shulpin_i_Jarvis_seq::generatePointsInCircle(center, radius, numPoints);
  std::vector<shulpin_i_Jarvis_seq::Point> expected = input;

  shulpin_i_Jarvis_seq::TestBody_Random_Circle(input, expected, numPoints);
}

TEST(shulpin_i_Jarvis_seqq, circle_r10_p10000) {
  shulpin_i_Jarvis_seq::Point center{0, 0};
  double radius = 10.0;
  size_t numPoints = 10000;
  std::vector<shulpin_i_Jarvis_seq::Point> input =
      shulpin_i_Jarvis_seq::generatePointsInCircle(center, radius, numPoints);
  std::vector<shulpin_i_Jarvis_seq::Point> expected = input;

  shulpin_i_Jarvis_seq::TestBody_Random_Circle(input, expected, numPoints);
}