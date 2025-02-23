#include <gtest/gtest.h>

#include <cstdint>
#include <ctime>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherSeq.hpp"

std::vector<double> GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_test_0) {
  int global_vector_size = 3;
  std::vector<double> global_vector = {5.69, -2.11, 0.52};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_test_1) {
  int global_vector_size = 9;
  std::vector<double> global_vector = {8, -2, 5, 10, 1, -7, 3, 12, -6};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_test_2) {
  int global_vector_size = 10;
  std::vector<double> global_vector = {-8.55,   1.85,   -4.0,   2.71828, 8.77,  -5.56562,
                                       -15.823, -6.971, 3.1415, 0.0,     3.1415};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_empty_test) {
  int global_vector_size = 0;
  std::vector<double> global_vector;
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_test_1) {
  int global_vector_size = 10;
  std::vector<double> global_vector = GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_test_2) {
  int global_vector_size = 50;
  std::vector<double> global_vector = GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_test_3) {
  int global_vector_size = 512;
  std::vector<double> global_vector = GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskData->inputs_count.emplace_back(global_vector.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskData->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential testTaskSequential(taskData);
  testTaskSequential.ValidationImpl();
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  for (int i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}