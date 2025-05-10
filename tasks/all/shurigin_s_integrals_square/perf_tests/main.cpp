#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "all/shurigin_s_integrals_square/include/ops_mpi.hpp"

namespace shurigin_s_integrals_square_mpi_perf_test {

static int GetMpiRank() {
  int rank_val = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_val);
  return rank_val;
}

TEST(ShuriginSIntegralsSquareMPI_Perf, TestPipelineRun) {
  const int rank = GetMpiRank();

  double down_limit_x = -1.0;
  double up_limit_x = 1.0;
  double down_limit_y = -1.0;
  double up_limit_y = 1.0;
  int count_x = 5000;
  int count_y = 5000;

  std::vector<double> input_data_vec;
  double result_val = 0.0;

  if (rank == 0) {
    input_data_vec = {
        down_limit_x, down_limit_y, up_limit_x, up_limit_y, static_cast<double>(count_x), static_cast<double>(count_y)};
  }

  auto f = [](const std::vector<double>& point) {
    double x = point[0];
    double y = point[1];
    return std::cos((x * x) + (y * y)) * (1 + (x * x) + (y * y));
  };

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 2;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.emplace_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_val));
    task_data->outputs_count.emplace_back(sizeof(double));
  } else {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(expected_input_bytes);
    task_data->outputs.emplace_back(nullptr);
    task_data->outputs_count.emplace_back(sizeof(double));
  }

  auto test_task = std::make_shared<shurigin_s_integrals_square_mpi::Integral>(task_data);
  test_task->SetFunction(f, 2);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    const double expected_result = 4.35751;
    const double epsilon = 1e-3;
    ASSERT_NEAR(result_val, expected_result, epsilon);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Perf, TestTaskRun) {
  const int rank = GetMpiRank();

  double down_limit_x = -1.0;
  double up_limit_x = 1.0;
  double down_limit_y = -1.0;
  double up_limit_y = 1.0;
  int count_x = 5000;
  int count_y = 5000;

  std::vector<double> input_data_vec;
  double result_val = 0.0;

  if (rank == 0) {
    input_data_vec = {
        down_limit_x, down_limit_y, up_limit_x, up_limit_y, static_cast<double>(count_x), static_cast<double>(count_y)};
  }

  auto f = [](const std::vector<double>& point) {
    double x = point[0];
    double y = point[1];
    return std::cos((x * x) + (y * y)) * (1 + (x * x) + (y * y));
  };

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 2;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.emplace_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_val));
    task_data->outputs_count.emplace_back(sizeof(double));
  } else {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(expected_input_bytes);
    task_data->outputs.emplace_back(nullptr);
    task_data->outputs_count.emplace_back(sizeof(double));
  }

  auto test_task = std::make_shared<shurigin_s_integrals_square_mpi::Integral>(task_data);
  test_task->SetFunction(f, 2);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);

  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    const double expected_result = 4.35751;
    const double epsilon = 1e-3;
    ASSERT_NEAR(result_val, expected_result, epsilon);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace shurigin_s_integrals_square_mpi_perf_test