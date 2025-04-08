#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/kolokolova_d_integral_simpson_method_tbb/include/ops_tbb.hpp"

TEST(kolokolova_d_integral_simpson_method_tbb, test1) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_tbb->inputs_count.emplace_back(step.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_tbb->inputs_count.emplace_back(bord.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_tbb->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb, func);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  double ans = 81.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_tbb, test2) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_tbb->inputs_count.emplace_back(step.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_tbb->inputs_count.emplace_back(bord.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_tbb->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb, func);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  double ans = 81.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_tbb, test3) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_tbb->inputs_count.emplace_back(step.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_tbb->inputs_count.emplace_back(bord.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_tbb->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb, func);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  double ans = 81.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_tbb, test4) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;

  // Create task_data
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_tbb->inputs_count.emplace_back(step.size());

  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_tbb->inputs_count.emplace_back(bord.size());

  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_tbb->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_tbb::TestTaskTBB test_task_tbb(task_data_tbb, func);
  ASSERT_EQ(test_task_tbb.Validation(), true);
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();
  double ans = 81.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}
