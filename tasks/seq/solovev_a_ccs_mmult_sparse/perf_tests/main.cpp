#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

TEST(solovev_a_ccs_mmult_sparse, test_pipeline_run) {
  solovev_a_matrix::MatrixInCCS_Sparse M1(2000000, 2000000);
  solovev_a_matrix::MatrixInCCS_Sparse M2(2000000, 1);
  solovev_a_matrix::MatrixInCCS_Sparse M3(2000000, 1);
  std::complex<double> vvector(2.0, 1.0);

  int l = 1;
  int m = 0;
  for (int i = 0; i <= 2000000; i++) {
    M1.col_p.push_back(m);
    m += l;
    l++;
  }

  l = 1;
  m = 0;
  for (int i = 0; i < M1.col_p[2000000]; i++) {
    M1.val.emplace_back(vvector);
    if (m >= l) {
      m = 0;
      l++;
    }
    M1.row.push_back(m);
    m++;
  }

  M2.col_p = {0, 2000000};
  for (int i = 0; i < 2000000; i++) {
    M2.val.emplace_back(vvector);
    M2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  auto test_task_sequential = std::make_shared<solovev_a_matrix::Seq_MatMultCCS>(task_data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
}

TEST(solovev_a_ccs_mmult_sparse, test_task_run) {
  solovev_a_matrix::MatrixInCCS_Sparse M1(2000000, 2000000);
  solovev_a_matrix::MatrixInCCS_Sparse M2(2000000, 1);
  solovev_a_matrix::MatrixInCCS_Sparse M3(2000000, 1);
  std::complex<double> vvector(2.0, 1.0);

  int l = 1;
  int m = 0;
  for (int i = 0; i <= 2000000; i++) {
    M1.col_p.emplace_back(m);
    m += l;
    l++;
  }

  l = 1;
  m = 0;
  for (int i = 0; i < M1.col_p[2000000]; i++) {
    M1.val.emplace_back(vvector);
    if (m >= l) {
      m = 0;
      l++;
    }
    M1.row.push_back(m);
    m++;
  }

  M2.col_p = {0, 2000000};
  for (int i = 0; i < 2000000; i++) {
    M2.val.emplace_back(vvector);
    M2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  auto test_task_sequential = std::make_shared<solovev_a_matrix::Seq_MatMultCCS>(task_data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
}
