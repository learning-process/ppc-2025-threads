#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kholin_k_multidimensional_integrals_rectangle/include/ops_seq.hpp"

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_validation) {
  // Create data
  double dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_pre_processing) {
  // Create data
  double dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_run) {
  // Create data
  double dim = 1.0;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, test_post_processing) {
  // Create data
  double dim = 1.0;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, single_integral_one_var) {
  // Create data
  double dim = 1.0;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 1e-2;
  double n = 20.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 0.46;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, single_integral_two_var) {
  // Create data
  double dim = 1.0;
  std::vector<double> values{0.0, 3.0};
  auto f = [](const std::vector<double> &f_values) { return std::exp(-f_values[0] + f_values[1]); };
  std::vector<double> in_lower_limits{-1};
  std::vector<double> in_upper_limits{5};
  double epsilon = 1e-1;
  double n = 40.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 54.4;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, double_integral_two_var) {
  // Create data
  double dim = 2.0;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return (f_values[0] * f_values[0]) + (f_values[1] * f_values[1]);
  };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double epsilon = 1e-3;
  double n = 47.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 913.333;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, double_integral_one_var) {
  // Create data
  double dim = 2.0;
  std::vector<double> values{-17.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return 289 + (f_values[1] * f_values[1]); };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double epsilon = 1e-1;
  double n = 7.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 6026.7;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, triple_integral_three_var) {
  // Create data
  double dim = 3.0;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + f_values[1] + f_values[2]; };
  std::vector<double> in_lower_limits{-4, 6, 7};
  std::vector<double> in_upper_limits{4, 13, 8};
  double epsilon = 1e-2;
  double n = 16.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 952;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, triple_integral_two_var) {
  // Create data
  double dim = 3.0;
  std::vector<double> values{0.0, 5.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + f_values[2]; };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double epsilon = 1e-2;
  double n = 39.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 9450;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_seq, triple_integral_one_var) {
  // Create data
  double dim = 3.0;
  std::vector<double> values{0.0, 5.0, -10.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + (-10.0); };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double epsilon = 1e-2;
  double n = 65.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(values.size());
  task_data_seq->inputs_count.emplace_back(in_lower_limits.size());
  task_data_seq->inputs_count.emplace_back(in_upper_limits.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_seq->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  ASSERT_EQ(test_task_sequential.PreProcessing(), true);
  ASSERT_EQ(test_task_sequential.Run(), true);
  ASSERT_EQ(test_task_sequential.PostProcessing(), true);

  double ref_i = 900;
  ASSERT_NEAR(ref_i, out_i[0], epsilon);
  delete f_object;
}