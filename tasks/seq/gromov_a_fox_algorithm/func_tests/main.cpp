    #include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/gromov_a_fox_algorithm/include/ops_seq.hpp"

TEST(gromov_a_fox_algorithm_seq, test_4x4) {
  constexpr size_t n = 4;

  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> B = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> out(n * n, 0.0);

  std::vector<double> expected = {80.0,  70.0,  60.0,  50.0,  240.0, 214.0, 188.0, 162.0,
                                  400.0, 358.0, 316.0, 274.0, 560.0, 502.0, 444.0, 386.0};

  std::vector<double> input;
  input.insert(input.end(), A.begin(), A.end());
  input.insert(input.end(), B.begin(), B.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_50x50) {
  constexpr size_t n = 50;

  std::vector<double> A(n * n, 0.0);
  std::vector<double> B(n * n, 0.0);
  std::vector<double> out(n * n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      A[i * n + j] = static_cast<double>(i * n + j + 1);
      B[i * n + j] = static_cast<double>(n * n - (i * n + j));
    }
  }

  std::vector<double> input;
  input.insert(input.end(), A.begin(), A.end());
  input.insert(input.end(), B.begin(), B.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<double> expected(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        expected[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}