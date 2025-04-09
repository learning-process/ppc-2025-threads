#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/karaseva_e_congrad_tbb/include/ops_tbb.hpp"

namespace {

// Function to generate a random symmetric positive-definite matrix of size matrix_size x matrix_size
// The matrix is computed as A = R^T * R
std::vector<double> GenerateRandomSPDMatrix(size_t matrix_size, unsigned int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> r_matrix(matrix_size * matrix_size);
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    r_matrix[i] = dist(gen);
  }
  std::vector<double> a_matrix(matrix_size * matrix_size, 0.0);
  // Compute a_matrix = R^T * R
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        a_matrix[(i * matrix_size) + j] += (r_matrix[(k * matrix_size) + i] * r_matrix[(k * matrix_size) + j]);
      }
    }
  }
  // Add diagonal dominance
  for (size_t i = 0; i < matrix_size; ++i) {
    a_matrix[(i * matrix_size) + i] += static_cast<double>(matrix_size);
  }
  return a_matrix;
}

// Helper function to multiply a_matrix (size matrix_size x matrix_size) by vector x (length matrix_size)
std::vector<double> MultiplyMatrixVector(const std::vector<double>& a_matrix, const std::vector<double>& x,
                                         size_t matrix_size) {
  std::vector<double> result(matrix_size, 0.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      result[i] += (a_matrix[(i * matrix_size) + j] * x[j]);
    }
  }
  return result;
}

}  // namespace

TEST(karaseva_e_congrad_tbb, test_identity) {
  constexpr size_t kN = 50;

  std::vector<double> a_matrix(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    a_matrix[(i * kN) + i] = 1.0;
  }

  std::vector<double> b(kN);
  for (size_t i = 0; i < kN; ++i) {
    b[i] = static_cast<double>(i + 1);
  }

  std::vector<double> x(kN, 0.0);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_tbb->inputs_count.push_back(kN * kN);
  task_data_tbb->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_tbb->inputs_count.push_back(kN);
  task_data_tbb->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_tbb->outputs_count.push_back(kN);

  karaseva_e_congrad_tbb::TestTaskTBB test_task_tbb(task_data_tbb);
  ASSERT_TRUE(test_task_tbb.Validation());
  test_task_tbb.PreProcessing();
  test_task_tbb.Run();
  test_task_tbb.PostProcessing();

  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], b[i], 1e-9);
  }
}