#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/anufriev_d_integrals_simpson/include/ops_seq.hpp"

TEST(anufriev_d_integrals_simpson_seq, test_x2_plus_y2) {
  std::vector<double> in = {0.0, 1.0, 100, 0.0, 1.0, 100, 0};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(task_data_seq);

  auto *in_ptr = new double[in.size()];
  memcpy(in_ptr, in.data(), in.size() * sizeof(double));

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  double result = out[0];
  ASSERT_NEAR(result, 2.0 / 3.0, 1e-3);

  delete[] in_ptr;
}

TEST(anufriev_d_integrals_simpson_seq, test_x2_plus_y2_x_not_equal_y) {
  std::vector<double> in = {0.0, 1.0, 100, 0.0, 1.0, 80, 0};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(task_data_seq);

  auto *in_ptr = new double[in.size()];
  memcpy(in_ptr, in.data(), in.size() * sizeof(double));

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  double result = out[0];
  ASSERT_NEAR(result, 2.0 / 3.0, 1e-3);

  delete[] in_ptr;
}

TEST(anufriev_d_integrals_simpson_seq, test_sin_cos_x_not_equal_y) {
  std::vector<double> in = {0.0, std::acos(-1) / 2.0, 100, 0.0, std::acos(-1) / 2.0, 80, 1};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(task_data_seq);

  auto *in_ptr = new double[in.size()];
  memcpy(in_ptr, in.data(), in.size() * sizeof(double));

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  double result = out[0];
  ASSERT_NEAR(result, 1.0, 1e-3);

  delete[] in_ptr;
}

TEST(anufriev_d_integrals_simpson_seq, test_sin_cos) {
  std::vector<double> in = {0.0, std::acos(-1) / 2.0, 100, 0.0, std::acos(-1) / 2.0, 100, 1};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  anufriev_d_integrals_simpson_seq::IntegralsSimpsonSequential task(task_data_seq);

  auto *in_ptr = new double[in.size()];
  memcpy(in_ptr, in.data(), in.size() * sizeof(double));

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  double result = out[0];
  ASSERT_NEAR(result, 1.0, 1e-3);

  delete[] in_ptr;
}