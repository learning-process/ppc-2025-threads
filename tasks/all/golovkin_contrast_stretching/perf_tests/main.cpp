// Golovkin Maksim
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/golovkin_contrast_stretching/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(golovkin_contrast_stretching_mpi, test_pipeline_run_mpi_omp) {
  constexpr size_t kCount = 1'000'000;
  std::vector<uint8_t> in(kCount);
  std::vector<uint8_t> out(kCount, 0);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (size_t i = 0; i < kCount; ++i) {
      in[i] = static_cast<uint8_t>(i % 256);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingMPI_OMP<uint8_t> task(task_data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    ASSERT_EQ(out.size(), in.size());
    EXPECT_GE(out[100], out[50]);

    auto [min_it, max_it] = std::minmax_element(in.begin(), in.end());
    uint8_t min_val = *min_it;
    uint8_t max_val = *max_it;
    if (min_val != max_val) {
      for (size_t i = 0; i < kCount; i += kCount / 100) {
        double expected = (in[i] - min_val) * 255.0 / (max_val - min_val);
        EXPECT_NEAR(out[i], expected, 1.0);
      }
    }
  }
}

TEST(golovkin_contrast_stretching_mpi, test_task_run_mpi_omp) {
  constexpr size_t kCount = 1'000'000;
  std::vector<uint8_t> in(kCount);
  std::vector<uint8_t> out(kCount, 0);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (size_t i = 0; i < kCount; ++i) {
      in[i] = static_cast<uint8_t>(i % 256);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingMPI_OMP<uint8_t> task(task_data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    ASSERT_EQ(out.size(), in.size());
    EXPECT_GE(out[100], out[50]);

    uint8_t out_min = *std::min_element(out.begin(), out.end());
    uint8_t out_max = *std::max_element(out.begin(), out.end());
    EXPECT_EQ(out_min, 0);
    EXPECT_EQ(out_max, 255);
  }
}