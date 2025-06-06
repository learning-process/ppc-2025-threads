// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/leontev_n_average/include/ops_seq.hpp"

namespace {
template <class InOutType>
void TaskEmplacement(std::shared_ptr<ppc::core::TaskData> &task_data_par, std::vector<InOutType> &global_vec,
                     std::vector<InOutType> &global_avg) {
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  task_data_par->inputs_count.emplace_back(global_vec.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_avg.data()));
  task_data_par->outputs_count.emplace_back(global_avg.size());
}
}  // namespace

TEST(leontev_n_average_seq, int_vector_avg) {
  // Create data
  std::vector<int32_t> in(5, 10);
  const int32_t expected_avg = 10;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement<int32_t>(task_data_seq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<int32_t> vec_avg_sequential(task_data_seq);
  ASSERT_TRUE(vec_avg_sequential.Validation());
  vec_avg_sequential.PreProcessing();
  vec_avg_sequential.Run();
  vec_avg_sequential.PostProcessing();
  ASSERT_EQ(expected_avg, out[0]);
}

TEST(leontev_n_average_seq, double_vector_avg) {
  // Create data
  std::vector<double> in(5, 10.0);
  const double expected_avg = 10.0;
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement<double>(task_data_seq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<double> vec_avg_sequential(task_data_seq);
  ASSERT_TRUE(vec_avg_sequential.Validation());
  vec_avg_sequential.PreProcessing();
  vec_avg_sequential.Run();
  vec_avg_sequential.PostProcessing();
  EXPECT_NEAR(out[0], expected_avg, 1e-6);
}

TEST(leontev_n_average_seq, float_vector_avg) {
  // Create data
  std::vector<float> in(5, 1.F);
  std::vector<float> out(1, 0.F);
  const float expected_avg = 1.F;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement<float>(task_data_seq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<float> vec_avg_sequential(task_data_seq);
  ASSERT_TRUE(vec_avg_sequential.Validation());
  vec_avg_sequential.PreProcessing();
  vec_avg_sequential.Run();
  vec_avg_sequential.PostProcessing();
  EXPECT_NEAR(out[0], expected_avg, 1e-3F);
}

TEST(leontev_n_average_seq, int32_vector_avg) {
  // Create data
  std::vector<int32_t> in(2000, 5);
  in[0] = 3;
  in[1] = 7;
  std::vector<int32_t> out(1, 0);
  const int32_t expected_avg = 5;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement<int32_t>(task_data_seq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<int32_t> vec_avg_sequential(task_data_seq);
  ASSERT_TRUE(vec_avg_sequential.Validation());
  vec_avg_sequential.PreProcessing();
  vec_avg_sequential.Run();
  vec_avg_sequential.PostProcessing();
  ASSERT_EQ(out[0], expected_avg);
}

TEST(leontev_n_average_seq, uint32_vector_avg) {
  // Create data
  std::vector<uint32_t> in(255, 2);
  in[0] = 0;
  in[1] = 4;
  std::vector<uint32_t> out(1, 0);
  const uint32_t expected_avg = 2;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement<uint32_t>(task_data_seq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<uint32_t> vec_avg_sequential(task_data_seq);
  ASSERT_TRUE(vec_avg_sequential.Validation());
  vec_avg_sequential.PreProcessing();
  vec_avg_sequential.Run();
  vec_avg_sequential.PostProcessing();
  ASSERT_EQ(out[0], expected_avg);
}

TEST(leontev_n_average_seq, vector_avg_0) {
  // Create data
  std::vector<int32_t> in(1, 0);
  const int32_t expected_avg = 0;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskEmplacement<int32_t>(task_data_seq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<int32_t> vec_avg_sequential(task_data_seq);
  ASSERT_TRUE(vec_avg_sequential.Validation());
  vec_avg_sequential.PreProcessing();
  vec_avg_sequential.Run();
  vec_avg_sequential.PostProcessing();
  ASSERT_EQ(expected_avg, out[0]);
}
