#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>
#include <iostream>

#include "core/task/include/task.hpp"
#include "stl/filateva_e_simpson/include/ops_stl.hpp"

TEST(filateva_e_simpson_stl, test_x_pow_2) {
  std::vector<double> param = {1, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return x * x * x / 3; };

  std::cerr << "\n" << res[0] << "\n";
  std::cerr << "\n" << integral_f(param[1]) - integral_f(param[0]) << "\n";
  std::cerr << "\n" << integral_f(param[1]) - integral_f(param[0]) - res[0] << "\n";

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_x_pow_2_negative) {
  std::vector<double> param = {-10, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return x * x * x / 3; };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_x) {
  std::vector<double> param = {1, 100, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return x * x / 2; };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_x_pow_3) {
  std::vector<double> param = {1, 100, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return x * x * x * x / 4; };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_x_del) {
  std::vector<double> param = {1, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return 1 / x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return std::log(x); };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_x_sin) {
  std::vector<double> param = {0, std::numbers::pi, 0.1};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return std::sin(x); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return -std::cos(x); };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_x_cos) {
  std::vector<double> param = {0, std::numbers::pi / 2, 0.1};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return std::cos(x); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) { return std::sin(x); };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_gausa) {
  std::vector<double> param = {0, 1, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return pow(std::numbers::e, -pow(x, 2)); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  ASSERT_NEAR(res[0], 0.746824, param[2]);
}

TEST(filateva_e_simpson_stl, test_sum_integral) {
  std::vector<double> param = {0, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return pow(x, 3) + pow(x, 2) + x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_stl::Func integral_f = [](double x) {
    return (pow(x, 4) / 4) + (pow(x, 3) / 3) + (pow(x, 2) / 2);
  };

  ASSERT_NEAR(res[0], integral_f(param[1]) - integral_f(param[0]), param[2]);
}

TEST(filateva_e_simpson_stl, test_error_1) {
  std::vector<double> param = {100, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}

TEST(filateva_e_simpson_stl, test_error_2) {
  std::vector<double> param = {0, 10, 20};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}

TEST(filateva_e_simpson_stl, test_error_3) {
  std::vector<double> param = {0, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}

TEST(filateva_e_simpson_stl, test_error_4) {
  std::vector<double> param = {0, 10, 0.001};
  std::vector<double> res(1, 0);
  filateva_e_simpson_stl::Func f = [](double x) { return x * x; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(param.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(2);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(2);

  filateva_e_simpson_stl::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}
