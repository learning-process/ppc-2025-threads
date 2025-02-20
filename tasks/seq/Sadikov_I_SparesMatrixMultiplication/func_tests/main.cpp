#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/Sadikov_I_SparesMatrixMultiplication/include/ops_seq.hpp"

TEST(Sadikov_I_SparseMatrixMultiplication_task_seq, test_rect_matrixes) {
  constexpr auto epsilon = 0.000001;
  std::vector<double> fmatrix{0, 0, 0, 5.0, 2.0, 0, 1.0, 0, 7.0, 7.0, 0, 0};
  std::vector<double> smatrix{1.0, 0, 0, 2.0, 0, 8.0, 0, 0, 0, 0, 5.0, 0};
  std::vector<double> out(9, 0.0);
  std::vector<double> testOut{0.0, 25.0, 0.0, 2.0, 0.0, 0.0, 21.0, 0.0, 56.0};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->inputs_count.emplace_back(4);
  task_data_seq->inputs_count.emplace_back(4);
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], testOut[i], epsilon);
  }
}

TEST(Sadikov_I_SparseMatrixMultiplication_task_seq, test_square_matrixes) {
  constexpr auto epsilon = 0.000001;
  std::vector<double> fmatrix{1.0, 0.0, 0.0, 0.0, 7.0, 0.0, 4.0, 9.0, 0.0};
  std::vector<double> smatrix{0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0};
  std::vector<double> out(9, 0.0);
  std::vector<double> testOut{0.0, 0.0, 3.0, 14.0, 0.0, 0.0, 18.0, 0.0, 12.0};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  for (auto i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(3);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], testOut[i], epsilon);
  }
}

TEST(Sadikov_I_SparseMatrixMultiplication_task_seq, test_empty_matrixes) {
  constexpr auto epsilon = 0.000001;
  std::vector<double> fmatrix;
  std::vector<double> smatrix;
  std::vector<double> out;
  std::vector<double> testOut;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  Sadikov_I_SparseMatrixMultiplication_task_seq::CCSMatrixSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], testOut[i], epsilon);
  }
}