#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/kondratev_ya_ccs_complex_multiplication/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
void FillMatrix(kondratev_ya_ccs_complex_multiplication_all::CCSMatrix &matrix, std::complex<double> value) {
  for (int i = 0; i < matrix.rows; i++) {
    matrix.values.emplace_back(value);
    matrix.row_index.emplace_back(i);
    matrix.col_ptrs[i] = i;
  }
  matrix.col_ptrs[matrix.rows] = matrix.rows;
}
}  // namespace

TEST(kondratev_ya_ccs_complex_multiplication_all, test_pipeline_run) {
  static boost::mpi::communicator world;
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_all::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_all::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_all::CCSMatrix c({kCount, kCount});

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    FillMatrix(a, {2.0, 1.0});
    FillMatrix(b, {3.0, 2.0});
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_all->inputs_count.emplace_back(2);

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
    task_data_all->outputs_count.emplace_back(1);
  }
  auto test_task_sequential = std::make_shared<kondratev_ya_ccs_complex_multiplication_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(kondratev_ya_ccs_complex_multiplication_all, test_task_run) {
  static boost::mpi::communicator world;
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_all::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_all::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_all::CCSMatrix c({kCount, kCount});

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    FillMatrix(a, {2.0, 1.0});
    FillMatrix(b, {3.0, 2.0});

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
    task_data_all->inputs_count.emplace_back(2);

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
    task_data_all->outputs_count.emplace_back(1);
  }
  auto test_task_alluential = std::make_shared<kondratev_ya_ccs_complex_multiplication_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
