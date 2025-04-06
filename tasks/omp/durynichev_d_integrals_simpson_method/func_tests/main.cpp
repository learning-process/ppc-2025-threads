#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_x_squared) {
  std::vector<double> in = {0.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  EXPECT_NEAR(out[0], 1.0 / 3.0, 1e-5);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_x_squared_wider_range) {
  std::vector<double> in = {0.0, 2.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  EXPECT_NEAR(out[0], 8.0 / 3.0, 1e-5);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_x2_plus_y2) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  EXPECT_NEAR(out[0], 2.0 / 3.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_x2_plus_y2_wider_range) {
  std::vector<double> in = {0.0, 2.0, 0.0, 2.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  EXPECT_NEAR(out[0], 16.0 / 3.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_x2_plus_y2_symmetric_range) {
  std::vector<double> in = {-1.0, 1.0, -1.0, 1.0, 100};
  std::vector<double> out(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  durynichev_d_integrals_simpson_method_omp::SimpsonIntegralOpenMP task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  EXPECT_NEAR(out[0], 8.0 / 3.0, 1e-4);
}