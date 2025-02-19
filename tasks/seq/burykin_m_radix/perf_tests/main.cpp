#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/burykin_m_radix/include/ops_seq.hpp"

namespace {

std::vector<int> GenerateRandomVector(size_t size, int min_val = -10000, int max_val = 10000) {
  std::vector<int> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min_val, max_val);
  for (auto &elem : vec) {
    elem = dis(gen);
  }
  return vec;
}

}  // namespace

TEST(burykin_m_radix_seq, test_pipeline_run) {
  constexpr size_t num_elements = 100000;

  // Создаём входной случайный вектор и вычисляем ожидаемый результат (отсортированный)
  std::vector<int> input = GenerateRandomVector(num_elements);
  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  // Выделяем память под результат (заполняем нулями)
  std::vector<int> output(num_elements, 0);

  // Создаём task_data и заполняем указатели на входные/выходные данные
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(output.size()));

  // Создаём задачу поразрядной сортировки
  auto task = std::make_shared<burykin_m_radix_seq::RadixSequential>(task_data);

  // Настраиваем параметры измерения производительности
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности полного конвейера
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // После полного конвейера в task_data->outputs обновлён результат сортировки
  EXPECT_EQ(output, expected);
}

TEST(burykin_m_radix_seq, test_task_run) {
  constexpr size_t num_elements = 100000;

  // Создаём входной случайный вектор и вычисляем ожидаемый результат (отсортированный)
  std::vector<int> input = GenerateRandomVector(num_elements);
  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());

  // Выделяем память под результат (заполняем нулями)
  std::vector<int> output(num_elements, 0);

  // Создаём task_data и заполняем указатели на входные/выходные данные
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(output.size()));

  // Создаём задачу поразрядной сортировки
  auto task = std::make_shared<burykin_m_radix_seq::RadixSequential>(task_data);

  // Необходимо выполнить предварительную обработку, чтобы внутренние структуры задачи были инициализированы
  ASSERT_TRUE(task->PreProcessing());

  // Настраиваем параметры измерения производительности для Run() функции
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности только функции Run()
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // После измерения Run() вызываем PostProcessing(), чтобы результат был скопирован в выходной буфер
  ASSERT_TRUE(task->PostProcessing());

  EXPECT_EQ(output, expected);
}
