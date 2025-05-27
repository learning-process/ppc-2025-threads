#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/gromov_a_fox_algorithm/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(gromov_a_fox_algorithm, Test_Matrix_Multiplication_Random_Integers) {
  boost::mpi::communicator world;
  constexpr size_t n = 6;
  std::vector<double> a(n * n);
  std::vector<double> b(n * n);
  std::vector<double> c(n * n, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-10, 10);

  if (world.rank() == 0) {
    for (size_t i = 0; i < n * n; ++i) {
      a[i] = static_cast<double>(dis(gen));
      b[i] = static_cast<double>(dis(gen));
    }
  }

  std::vector<double> input;
  if (world.rank() == 0) {
    input.insert(input.end(), a.begin(), a.end());
    input.insert(input.end(), b.begin(), b.end());
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_all->inputs_count.emplace_back(input.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data_all->outputs_count.emplace_back(n * n);
  }

  gromov_a_fox_algorithm_all::TestTaskAll matrix_multiplication(task_data_all);

  ASSERT_TRUE(matrix_multiplication.ValidationImpl());
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> expected(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
          expected[i * n + j] += a[i * n + k] * b[k * n + j];
        }
      }
    }
    for (size_t i = 0; i < n * n; ++i) {
      EXPECT_EQ(c[i], expected[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(gromov_a_fox_algorithm, Test_Matrix_Multiplication_Identity) {
  boost::mpi::communicator world;
  constexpr size_t n = 3;
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> c(n * n, 0);

  std::vector<double> input;
  if (world.rank() == 0) {
    input.insert(input.end(), a.begin(), a.end());
    input.insert(input.end(), b.begin(), b.end());
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_all->inputs_count.emplace_back(input.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data_all->outputs_count.emplace_back(n * n);
  }

  gromov_a_fox_algorithm_all::TestTaskAll matrix_multiplication(task_data_all);

  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(c, a);
  }
}

TEST(gromov_a_fox_algorithm, Test_Matrix_Multiplication_Single_Element) {
  boost::mpi::communicator world;
  constexpr size_t n = 1;
  std::vector<double> a = {5.0};
  std::vector<double> b = {3.0};
  std::vector<double> c(n * n, 0);
  std::vector<double> expected = {15.0};

  std::vector<double> input;
  if (world.rank() == 0) {
    input.insert(input.end(), a.begin(), a.end());
    input.insert(input.end(), b.begin(), b.end());
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_all->inputs_count.emplace_back(input.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data_all->outputs_count.emplace_back(n * n);
  }

  gromov_a_fox_algorithm_all::TestTaskAll matrix_multiplication(task_data_all);

  ASSERT_TRUE(matrix_multiplication.ValidationImpl());
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(c, expected);
  }
}

TEST(gromov_a_fox_algorithm, Test_Matrix_Multiplication_Zero) {
  boost::mpi::communicator world;
  constexpr size_t n = 3;
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b(n * n, 0.0);
  std::vector<double> c(n * n, 0);
  std::vector<double> expected(n * n, 0.0);
  std::vector<double> input;
  if (world.rank() == 0) {
    input.insert(input.end(), a.begin(), a.end());
    input.insert(input.end(), b.begin(), b.end());
  }

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_all->inputs_count.emplace_back(input.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data_all->outputs_count.emplace_back(n * n);
  }

  gromov_a_fox_algorithm_all::TestTaskAll matrix_multiplication(task_data_all);

  ASSERT_TRUE(matrix_multiplication.ValidationImpl());
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(c, expected);
  }
}