#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/durynichev_d_integrals_simpson_method/include/ops_omp.hpp"

// Тесты для квадратичной функции (оригинальные тесты)
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_x_squared) {
  std::vector<double> in = {0.0, 1.0, 100, 0};
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

// Тест для синуса
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_sin) {
  // Интеграл sin(x) от 0 до π должен быть равен 2
  std::vector<double> in = {0.0, M_PI, 1000, 1}; // 1 - индекс для Sin
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
  // Интеграл sin(x) от 0 до π равен 2.0
  EXPECT_NEAR(out[0], 2.0, 1e-4);
}

// Тест для косинуса
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_cos) {
  // Интеграл cos(x) от 0 до π/2 должен быть равен 1
  std::vector<double> in = {0.0, M_PI_2, 1000, 2}; // 2 - индекс для Cos
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
  // Интеграл cos(x) от 0 до π/2 равен 1.0
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

// Тест для экспоненты с меньшим диапазоном
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_exp) {
  // Интеграл e^x от 0 до 1 должен быть равен e-1
  std::vector<double> in = {0.0, 1.0, 1000, 3}; // 3 - индекс для Exp
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
  // Интеграл e^x от 0 до 1 равен e-1 ≈ 1.718
  EXPECT_NEAR(out[0], std::exp(1.0) - 1.0, 1e-4);
}

// Тест для натурального логарифма
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_log) {
  // Интеграл ln(x) от 1 до e должен быть равен 1
  std::vector<double> in = {1.0, std::exp(1.0), 1000, 4}; // 4 - индекс для Log
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
  // Интеграл ln(x) от 1 до e равен 1.0
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

// Тест для комбинированной функции с исправленным ожидаемым значением
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_combined) {
  // Интеграл (sin(x) + cos(x) + x^2) от 0 до 1
  std::vector<double> in = {0.0, 1.0, 1000, 5}; // 5 - индекс для Combined
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
  // Аналитический результат: (sin(1) - sin(0)) + (sin(1) - 0) + 1/3
  double expected = std::sin(1.0) + (1.0 - std::cos(1.0)) + (1.0 / 3.0);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// 2D тесты с разными функциями
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_sin) {
  // Интеграл sin(x)*sin(y) от 0 до π/2 в обоих измерениях (уменьшаем диапазон для точности)
  std::vector<double> in = {0.0, M_PI_2, 0.0, M_PI_2, 100, 1}; // 1 - индекс для Sin
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
  // Интеграл sin(x)*sin(y) от 0 до π/2 в обоих измерениях равен (1 - cos(π/2))^2 = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_cos) {
  // Интеграл cos(x)*cos(y) от 0 до π/2 в обоих измерениях
  std::vector<double> in = {0.0, M_PI_2, 0.0, M_PI_2, 100, 2}; // 2 - индекс для Cos
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
  // Интеграл cos(x)*cos(y) от 0 до π/2 в обоих измерениях равен sin(π/2)*sin(π/2) = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_exp) {
  // Интеграл e^(x+y) от 0 до 1 в обоих измерениях
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 100, 3}; // 3 - индекс для Exp
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
  // Аналитический результат: (e-1)^2
  double expected = (std::exp(1.0) - 1.0) * (std::exp(1.0) - 1.0);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_combined) {
  // Интеграл (sin(x) + cos(y) + x^2 + y^2) от 0 до 1 в обоих измерениях
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 100, 5}; // 5 - индекс для Combined
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

  // Интеграл (sin(x) + cos(y) + x^2 + y^2) от 0 до 1 в обоих измерениях
  // = (1-cos(1)) * 1 + sin(1) * 1 + 1/3 * 1 + 1 * 1/3
  double expected = (1.0 - std::cos(1.0)) + std::sin(1.0) + (1.0/3.0) + (1.0/3.0);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// Тест для 3D интеграла квадратичной функции
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_x_squared) {
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 0};
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
  // Для интеграла x^2 + y^2 + z^2 от [0,1]^3 ожидаемый результат = 1.0
  EXPECT_NEAR(out[0], 1.0, 1e-5);
}

// Тест для синуса в 3D
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_sin) {
  // Интеграл sin(x)*sin(y)*sin(z) от 0 до π/2 в трех измерениях
  std::vector<double> in = {0.0, M_PI_2, 0.0, M_PI_2, 0.0, M_PI_2, 100, 1}; // 1 - индекс для Sin
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
  // Интеграл sin(x)*sin(y)*sin(z) от 0 до π/2 в трех измерениях равен (1-cos(π/2))^3 = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

// Тест для косинуса в 3D
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_cos) {
  // Интеграл cos(x)*cos(y)*cos(z) от 0 до π/2 в трех измерениях
  std::vector<double> in = {0.0, M_PI_2, 0.0, M_PI_2, 0.0, M_PI_2, 100, 2}; // 2 - индекс для Cos
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
  // Интеграл cos(x)*cos(y)*cos(z) от 0 до π/2 в трех измерениях равен sin(π/2)^3 = 1
  EXPECT_NEAR(out[0], 1.0, 1e-4);
}

// Тест для экспоненты в 3D
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_exp) {
  // Интеграл e^(x+y+z) от 0 до 1 в трех измерениях
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 3}; // 3 - индекс для Exp
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
  // Интеграл e^(x+y+z) от 0 до 1 в трех измерениях равен (e-1)^3
  double expected = std::pow(std::exp(1.0) - 1.0, 3);
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// Тест для комбинированной функции в 3D
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_combined) {
  // Интеграл (sin(x) + cos(y) + sin(z) + x^2 + y^2 + z^2) от 0 до 1 в трех измерениях
  std::vector<double> in = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 100, 5}; // 5 - индекс для Combined
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

  // Интеграл (sin(x) + cos(y) + sin(z) + x^2 + y^2 + z^2) от 0 до 1 в трех измерениях
  // = (1-cos(1)) + sin(1) + (1-cos(1)) + 1/3 + 1/3 + 1/3
  double expected = (1.0 - std::cos(1.0)) + std::sin(1.0) + (1.0 - std::cos(1.0)) + 1.0;
  EXPECT_NEAR(out[0], expected, 1e-4);
}

// Рандомный тест для 1D интеграла
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_1D_random) {
  // Рандомные границы интегрирования
  double a = -10.0 + (rand() % 200) / 10.0;  // от -10 до 10
  double b = a + (rand() % 100) / 10.0;      // от a до a+10
  int n = 100 + rand() % 1000;               // от 100 до 1100 шагов
  n = n - (n % 2);                           // Делаем n четным

  std::vector<double> in = {a, b, static_cast<double>(n), 0}; // Квадратичная функция
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

  // Аналитический результат для x^2 на отрезке [a,b]
  double expected = (b*b*b - a*a*a) / 3.0;
  EXPECT_NEAR(out[0], expected, 1e-3);
}

// Рандомный тест для 2D интеграла
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_2D_random) {
  // Рандомные границы интегрирования
  double x_a = -5.0 + (rand() % 100) / 10.0;  // от -5 до 5
  double x_b = x_a + (rand() % 50) / 10.0;    // от x_a до x_a+5
  double y_a = -5.0 + (rand() % 100) / 10.0;  // от -5 до 5
  double y_b = y_a + (rand() % 50) / 10.0;    // от y_a до y_a+5
  int n = 100 + rand() % 200;                 // от 100 до 300 шагов
  n = n - (n % 2);                           // Делаем n четным

  std::vector<double> in = {x_a, x_b, y_a, y_b, static_cast<double>(n), 0}; // Квадратичная функция
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

  // Аналитический результат для x^2 + y^2 на прямоугольнике [x_a,x_b]x[y_a,y_b]
  double expected_x = (x_b*x_b*x_b - x_a*x_a*x_a) / 3.0;
  double expected_y = (y_b*y_b*y_b - y_a*y_a*y_a) / 3.0;
  double expected = expected_x * (y_b - y_a) + expected_y * (x_b - x_a);
  EXPECT_NEAR(out[0], expected, 1e-3);
}

// Рандомный тест для 3D интеграла
TEST(durynichev_d_integrals_simpson_method_omp, test_integral_3D_random) {
  // Рандомные границы интегрирования
  double x_a = -2.0 + (rand() % 40) / 10.0;   // от -2 до 2
  double x_b = x_a + (rand() % 20) / 10.0;    // от x_a до x_a+2
  double y_a = -2.0 + (rand() % 40) / 10.0;   // от -2 до 2
  double y_b = y_a + (rand() % 20) / 10.0;    // от y_a до y_a+2
  double z_a = -2.0 + (rand() % 40) / 10.0;   // от -2 до 2
  double z_b = z_a + (rand() % 20) / 10.0;    // от z_a до z_a+2
  int n = 50 + rand() % 100;                  // от 50 до 150 шагов
  n = n - (n % 2);                           // Делаем n четным

  std::vector<double> in = {x_a, x_b, y_a, y_b, z_a, z_b, static_cast<double>(n), 0}; // Квадратичная функция
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

  // Аналитический результат для x^2 + y^2 + z^2 на параллелепипеде [x_a,x_b]x[y_a,y_b]x[z_a,z_b]
  double expected_x = (x_b*x_b*x_b - x_a*x_a*x_a) / 3.0;
  double expected_y = (y_b*y_b*y_b - y_a*y_a*y_a) / 3.0;
  double expected_z = (z_b*z_b*z_b - z_a*z_a*z_a) / 3.0;
  double expected = expected_x * (y_b - y_a) * (z_b - z_a) +
                    expected_y * (x_b - x_a) * (z_b - z_a) +
                    expected_z * (x_b - x_a) * (y_b - y_a);
  EXPECT_NEAR(out[0], expected, 1e-3);
}