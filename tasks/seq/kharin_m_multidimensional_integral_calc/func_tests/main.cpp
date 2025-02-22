#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "include/ops_seq.hpp"

// Тест для интегрирования 3x3-сетки.
// В данном случае входные данные – значения функции, расположенные на равномерной квадратной сетке,
// и интеграл считается как сумма всех ячеек (при шаге равном 1).
// Для 3x3 сетки сумма равна 1+2+...+9 = 45.
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_3x3) {
  constexpr size_t kDim = 3;
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // Сумма = 45

  std::vector<int> out(1, 0);  // Результат – одно число (интеграл)
  std::vector<int> expected_out = {45};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

// Тест для интегрирования 5x5-сетки с последовательными значениями от 1 до 25.
// Сумма чисел от 1 до 25 равна 325.
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_5x5) {
  constexpr size_t kDim = 5;
  std::vector<int> in;
  for (int i = 1; i <= 25; ++i) {
    in.push_back(i);
  }  // Сумма = 325

  std::vector<int> out(1, 0);
  std::vector<int> expected_out = {325};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

// Тест с постоянными данными: 5x5-сетка, заполненная числом 3.
// Ожидаемый интеграл: 25 * 3 = 75.
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_const_data) {
  constexpr size_t kDim = 5;
  std::vector<int> in(kDim * kDim, 3);
  std::vector<int> out(1, 0);
  std::vector<int> expected_out = {75};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

// Тест с отрицательными данными: 5x5-сетка, содержащая значения от -1 до -25.
// Сумма чисел от 1 до 25 равна 325, следовательно, ожидаемый интеграл = -325.
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_negative_data) {
  constexpr size_t kDim = 5;
  std::vector<int> in;
  for (int i = 1; i <= 25; ++i) {
    in.push_back(-i);
  }  // Сумма = -325

  std::vector<int> out(1, 0);
  std::vector<int> expected_out = {-325};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

// Тест для большого объёма данных: 500x500-сетка, заполненная единицами.
// Ожидаемый интеграл = 500 * 500 = 250000.
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_large_data) {
  constexpr size_t kDim = 500;
  std::vector<int> in(kDim * kDim, 1);
  std::vector<int> out(1, 0);
  std::vector<int> expected_out = {static_cast<int>(kDim * kDim)};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}