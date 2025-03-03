#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

namespace {

std::vector<int> GenBinVec(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> bin_vec(size);

  for (int i = 0; i < size; i++) {
    bin_vec[i] = static_cast<int>(gen() % 2);
  }

  return bin_vec;
}

}  // namespace

TEST(voroshilov_v_convex_hull_components_seq, chc_pipeline_run) {
  std::vector<int> pixels = GenBinVec(10'000'000);
  int height = 10'000;
  int width = 1'000;

  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  auto chc_task_sequential =
      std::make_shared<voroshilov_v_convex_hull_components_seq::ChcTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(chc_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(voroshilov_v_convex_hull_components_seq, chc_task_run) {
  std::vector<int> pixels = GenBinVec(10'000'000);
  int height = 10'000;
  int width = 1'000;

  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  auto chc_task_sequential =
      std::make_shared<voroshilov_v_convex_hull_components_seq::ChcTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(chc_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
