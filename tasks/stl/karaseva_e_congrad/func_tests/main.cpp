#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/karaseva_e_congrad/include/ops_stl.hpp"

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
std::vector<double> MultiplyMatrixVector(const std::vector<double> &a_matrix, const std::vector<double> &x,
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

TEST(karaseva_a_test_task_stl, test_cg_solution_accuracy) {
  constexpr size_t kSize = 50;
  constexpr double kTolerance = 1e-6;

  // Generate SPD matrix and exact solution
  auto a_matrix = GenerateRandomSPDMatrix(kSize);
  std::vector<double> x_expected(kSize, 1.0);  // Expected solution vector (all ones)
  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);

  // Create buffers for data
  std::vector<double> solution(kSize, 0.0);

  // Create task data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_stl->inputs_count.emplace_back(a_matrix.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_vector.data()));
  task_data_stl->inputs_count.emplace_back(b_vector.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  task_data_stl->outputs_count.emplace_back(solution.size());

  // Create and run task
  karaseva_a_test_task_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  // Verify solution accuracy
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance)
        << "Mismatch at index " << i << ". Expected: " << x_expected[i] << ", Actual: " << solution[i];
  }
}

TEST(karaseva_a_test_task_stl, test_small_matrix_2x2) {
  constexpr size_t kSize = 2;
  constexpr double kTolerance = 1e-10;

  std::vector<double> a_matrix = {4.0, 1.0, 1.0, 3.0};
  std::vector<double> x_expected = {1.0, -2.0};
  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);

  std::vector<double> solution(kSize, 0.0);

  // Create task data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_stl->inputs_count.emplace_back(a_matrix.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_vector.data()));
  task_data_stl->inputs_count.emplace_back(b_vector.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  task_data_stl->outputs_count.emplace_back(solution.size());

  // Create and run task
  karaseva_a_test_task_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance) << "Mismatch at index " << i;
  }
}

TEST(karaseva_a_test_task_stl, test_zero_rhs) {
  constexpr size_t kSize = 50;
  constexpr double kTolerance = 1e-10;

  auto a_matrix = GenerateRandomSPDMatrix(kSize);
  std::vector<double> x_expected(kSize, 0.0);
  std::vector<double> b_vector(kSize, 0.0);

  std::vector<double> solution(kSize, 1.0);

  // Create task data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_stl->inputs_count.emplace_back(a_matrix.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_vector.data()));
  task_data_stl->inputs_count.emplace_back(b_vector.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  task_data_stl->outputs_count.emplace_back(solution.size());

  // Create and run task
  karaseva_a_test_task_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance) << "Non-zero solution at index " << i;
  }
}

TEST(karaseva_a_test_task_stl, test_random_solution) {
  constexpr size_t kSize = 30;
  constexpr double kTolerance = 1e-6;
  std::mt19937 gen(777);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  auto a_matrix = GenerateRandomSPDMatrix(kSize, gen());
  std::vector<double> x_expected(kSize);
  for (auto &val : x_expected) {
    val = dist(gen);
  }

  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);
  std::vector<double> solution(kSize, 0.0);

  // Create task data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_stl->inputs_count.emplace_back(a_matrix.size());
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_vector.data()));
  task_data_stl->inputs_count.emplace_back(b_vector.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  task_data_stl->outputs_count.emplace_back(solution.size());

  // Create and run task
  karaseva_a_test_task_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_TRUE(test_task_stl.Validation());
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance) << "Mismatch at index " << i;
  }
}