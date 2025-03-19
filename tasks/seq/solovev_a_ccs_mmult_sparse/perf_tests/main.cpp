#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

TEST(solovev_a_ccs_mmult_sparse_seq, test_pipeline_run) {
  solovev_a_matrix::MatrixInCcsSparse m1(2000000, 2000000);
  solovev_a_matrix::MatrixInCcsSparse m2(2000000, 1);
  solovev_a_matrix::MatrixInCcsSparse m3(2000000, 1);
  std::complex<double> vvector(2.0, 1.0);

  int l = 1;
  int m = 0;
  for (int i = 0; i <= 2000000; i++) {
    m1.col_p.push_back(m);
    m += l;
    l++;
  }

  l = 1;
  m = 0;
  for (int i = 0; i < m1.col_p[2000000]; i++) {
    m1.val.emplace_back(vvector);
    if (m >= l) {
      m = 0;
      l++;
    }
    m1.row.push_back(m);
    m++;
  }

  m2.col_p = {0, 2000000};
  for (int i = 0; i < 2000000; i++) {
    m2.val.emplace_back(vvector);
    m2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  auto test_task_sequential = std::make_shared<solovev_a_matrix::SeqMatMultCcs>(task_data_seq);

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
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(solovev_a_ccs_mmult_sparse_seq, test_task_run) {
  solovev_a_matrix::MatrixInCcsSparse m1(2000000, 2000000);
  solovev_a_matrix::MatrixInCcsSparse m2(2000000, 1);
  solovev_a_matrix::MatrixInCcsSparse m3(2000000, 1);
  std::complex<double> vvector(2.0, 1.0);

  int l = 1;
  int m = 0;
  for (int i = 0; i <= 2000000; i++) {
    m1.col_p.emplace_back(m);
    m += l;
    l++;
  }

  l = 1;
  m = 0;
  for (int i = 0; i < m1.col_p[2000000]; i++) {
    m1.val.emplace_back(vvector);
    if (m >= l) {
      m = 0;
      l++;
    }
    m1.row.push_back(m);
    m++;
  }

  m2.col_p = {0, 2000000};
  for (int i = 0; i < 2000000; i++) {
    m2.val.emplace_back(vvector);
    m2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  auto test_task_sequential = std::make_shared<solovev_a_matrix::SeqMatMultCcs>(task_data_seq);

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
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
