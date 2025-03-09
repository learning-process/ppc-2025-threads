// Copyright 2025 Dmitry Kalinin
#include <gtest/gtest.h>

#include <vector>

#include "seq/kalinin_d_jarvis_convex_hull/include/ops_seq.hpp"

TEST(kalinin_d_jarvis_convex_hull_seq, Empty_Input) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> resHull(hull.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  task_data_seq->outputs_count.emplace_back(resHull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Single_Point) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{0, 0}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{0, 0}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> resHull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  task_data_seq->outputs_count.emplace_back(resHull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(resHull[0], hull[0]);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Two_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{0, 0}, {1, 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{0, 0}, {1, 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> resHull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  task_data_seq->outputs_count.emplace_back(resHull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(resHull[0], hull[0]);
  ASSERT_EQ(resHull[1], hull[1]);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Duplicate_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{0, 0}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {2, 2}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{0, 0}, {2, 2}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> resHull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  task_data_seq->outputs_count.emplace_back(resHull.size());
  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(resHull[0], hull[0]);
  ASSERT_EQ(resHull[1], hull[1]);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Random_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{1, 4}, {3, 8}, {8, 2}, {5, 5}, {9, 1}, {4, 7}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{1, 4}, {3, 8}, {4, 7}, {9, 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> resHull(hull.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  task_data_seq->outputs_count.emplace_back(resHull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_seq, Rectangle_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{0, 0}, {0, 5}, {5, 5}, {5, 0},
                                                                 {1, 1}, {1, 4}, {4, 4}, {4, 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> resHull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  task_data_seq->outputs_count.emplace_back(resHull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}