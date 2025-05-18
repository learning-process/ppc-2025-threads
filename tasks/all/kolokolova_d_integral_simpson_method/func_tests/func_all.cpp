#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/kolokolova_d_integral_simpson_method/include/ops_all.hpp"
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

TEST(kolokolova_d_integral_simpson_method_all, test_matmul_50) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 81.0;
  double error = 1e-5;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_matmul_2) {
  auto func = [](std::vector<double> vec) { return 3 * vec[0] * vec[0] * vec[1] * vec[1]; };
  std::vector<int> step = {10, 10};
  std::vector<int> bord = {4, 6, 3, 6};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 9576.0;
  double error = 1e-5;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_matmul_3) {
  auto func = [](std::vector<double> vec) { return (2 * vec[2]) + (vec[1] / 5) + (4 * vec[0]); };
  std::vector<int> step = {30, 30, 30};
  std::vector<int> bord = {0, 1, 0, 3, 0, 2};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 25.7;
  double error = 0.1;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}