#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/yasakova_t_sparse_matrix_multiplication_omp/include/ops_omp.hpp"

namespace {
MatrixStructure RandMatrix(uint32_t num_rows, uint32_t num_cols, double percentage) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> distr(-10000, 10000);
  MatrixStructure result{
      .num_rows = num_rows, .num_cols = num_cols, .elements = std::vector<std::complex<double>>(num_rows * num_cols)};
  std::ranges::generate(result.elements, [&]() {
    const auto el = distr(gen);
    const auto re = (el < (distr.min() + ((distr.max() - distr.min()) * percentage))) ? el : 0;

    std::complex<double> cmplx;
    cmplx.real(re);
    if (re != 0.0) {
      cmplx.imag(distr(gen));
    }

    return cmplx;
  });
  return result;
}
}  // namespace

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_pipeline_run) {
  auto mat_a = RandMatrix(730, 730, 0.22);
  auto mat_b = RandMatrix(730, 730, 0.22);

  SparseMatrixFormat crs_lhs = ConvertToCRS(mat_a);
  SparseMatrixFormat crs_rhs = ConvertToCRS(mat_b);
  SparseMatrixFormat crs_out;

  auto elements = std::make_shared<ppc::core::TaskData>();
  elements->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  elements->inputs_count = {mat_a.num_rows, mat_a.num_cols, mat_b.num_rows, mat_b.num_cols};
  elements->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  elements->outputs_count = {1};

  auto task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(elements);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_EQ(ConvertFromCRS(crs_out), MatrixMultiply(mat_a, mat_b));
}

TEST(yasakova_t_sparse_matrix_multiplication_omp, test_task_run) {
  auto mat_a = RandMatrix(730, 730, 0.22);
  auto mat_b = RandMatrix(730, 730, 0.22);

  SparseMatrixFormat crs_lhs = ConvertToCRS(mat_a);
  SparseMatrixFormat crs_rhs = ConvertToCRS(mat_b);
  SparseMatrixFormat crs_out;

  auto elements = std::make_shared<ppc::core::TaskData>();
  elements->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  elements->inputs_count = {mat_a.num_rows, mat_a.num_cols, mat_b.num_rows, mat_b.num_cols};
  elements->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  elements->outputs_count = {1};

  auto task = std::make_shared<yasakova_t_sparse_matrix_multiplication_omp::SparseMatrixMultiplier>(elements);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_EQ(ConvertFromCRS(crs_out), MatrixMultiply(mat_a, mat_b));
}