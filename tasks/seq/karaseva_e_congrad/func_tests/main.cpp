#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/karaseva_e_congrad/include/ops_seq.hpp"

namespace {
// Function to generate a random symmetric positive definite matrix of size n x n
// The matrix is computed as A = R^T * R
std::vector<double> GenerateRandomSPDMatrix(size_t n, unsigned int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> R(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    R[i] = dist(gen);
  }
  std::vector<double> A(n * n, 0.0);
  // Compute A = R^T * R
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        A[i * n + j] += R[k * n + i] * R[k * n + j];
      }
    }
  }
  // Add diagonal dominance
  for (size_t i = 0; i < n; ++i) {
    A[i * n + i] += n;
  }
  return A;
}

// Helper function to multiply matrix A (size n x n) by vector x (length n)
std::vector<double> MultiplyMatrixVector(const std::vector<double>& A, const std::vector<double>& x, size_t n) {
  std::vector<double> result(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      result[i] += A[i * n + j] * x[j];
    }
  }
  return result;
}
}  // namespace

TEST(karaseva_e_congrad_seq, test_identity_50) {
  constexpr size_t n = 50;

  // Create identity matrix A of size n x n
  std::vector<double> A(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    A[i * n + i] = 1.0;
  }

  // Create vector b with elements 1.0, 2.0, ..., n
  std::vector<double> b(n);
  for (size_t i = 0; i < n; ++i) {
    b[i] = static_cast<double>(i + 1);
  }

  // Vector for the solution x, initially filled with zeros
  std::vector<double> x(n, 0.0);

  // Set up task data structure
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  // Create task
  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check that the computed solution x matches vector b within tolerance
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], b[i], 1e-9);
  }
}

TEST(karaseva_e_congrad_seq, test_random_spd_small) {
  constexpr size_t n = 20;  // system size

  // Generate a random SPD matrix A using a fixed seed
  auto A = GenerateRandomSPDMatrix(n, 42);

  // Generate a random true solution vector x_true
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> x_true(n);
  for (size_t i = 0; i < n; ++i) {
    x_true[i] = dist(gen);
  }

  // Compute right-hand side vector b = A * x_true
  auto b = MultiplyMatrixVector(A, x_true, n);

  // Vector for the computed solution x, initially zeros
  std::vector<double> x(n, 0.0);

  // Set up task data structure
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  // Create task
  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check that the computed solution x is close to the true solution x_true
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], x_true[i], 1e-6);
  }
}

TEST(karaseva_e_congrad_seq, test_random_spd_medium) {
  constexpr size_t n = 50;

  // Generate a random SPD matrix A using a different fixed seed
  auto A = GenerateRandomSPDMatrix(n, 123);

  // Generate a random true solution vector x_true
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> x_true(n);
  for (size_t i = 0; i < n; ++i) {
    x_true[i] = dist(gen);
  }

  // Compute right-hand side vector b = A * x_true
  auto b = MultiplyMatrixVector(A, x_true, n);

  // Vector for the computed solution x, initially zeros
  std::vector<double> x(n, 0.0);

  // Set up task data structure
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  // Create task
  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check that the computed solution x is close to the true solution x_true
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], x_true[i], 1e-6);
  }
}

TEST(karaseva_e_congrad_seq, test_random_spd_diagonal) {
  constexpr size_t n = 30;

  // Create a random diagonal SPD matrix A
  std::vector<double> A(n * n, 0.0);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist_diag(1.0, 10.0);
  for (size_t i = 0; i < n; ++i) {
    A[i * n + i] = dist_diag(gen);
  }

  // Generate a random true solution vector x_true
  std::vector<double> x_true(n);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  for (size_t i = 0; i < n; ++i) {
    x_true[i] = dist(gen);
  }

  // Compute right-hand side vector b = A * x_true
  auto b = MultiplyMatrixVector(A, x_true, n);

  // Vector for the computed solution x, initially zeros
  std::vector<double> x(n, 0.0);

  // Set up task data structure
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  // Create task
  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check that the computed solution x is close to the true solution x_true
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], x_true[i], 1e-6);
  }
}