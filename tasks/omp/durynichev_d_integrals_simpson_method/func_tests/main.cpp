#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <random>

#include "core/task/include/task.hpp"
#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

double RandomDouble(double min, double max, std::mt19937& gen) {
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}

int RandomEvenInt(int min, int max, std::mt19937& gen) {
  std::uniform_int_distribution<> dis(min / 2, max / 2);
  return dis(gen) * 2;
}

TEST(durynichev_d_integrals_simpson_method_omp, test_random_1D_x_squared_positive_range) {
  std::mt19937 gen(42);
  double a = RandomDouble(0.0, 5.0, gen);
  double b = RandomDouble(a, 10.0, gen);
  int n = RandomEvenInt(50, 200, gen);

  std::vector<double> in = {a, b, static_cast<double>(n)};
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

  double expected = (b * b * b - a * a * a) / 3.0;
  EXPECT_NEAR(out[0], expected, 1e-5);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_random_2D_x2_plus_y2_positive_range) {
  std::mt19937 gen(42);
  double x0 = RandomDouble(0.0, 5.0, gen);
  double x1 = RandomDouble(x0, 10.0, gen);
  double y0 = RandomDouble(0.0, 5.0, gen);
  double y1 = RandomDouble(y0, 10.0, gen);
  int n = RandomEvenInt(50, 200, gen);

  std::vector<double> in = {x0, x1, y0, y1, static_cast<double>(n)};
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

  double expected = ((x1 * x1 * x1 - x0 * x0 * x0) * (y1 - y0) + (y1 * y1 * y1 - y0 * y0 * y0) * (x1 - x0)) / 3.0;
  EXPECT_NEAR(out[0], expected, 1e-4);
}

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